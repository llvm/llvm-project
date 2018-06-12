//===- afl_driver.cpp - a glue between AFL and libFuzzer --------*- C++ -* ===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//===----------------------------------------------------------------------===//

/* This file allows to fuzz libFuzzer-style target functions
 (LLVMFuzzerTestOneInput) with AFL using AFL's persistent (in-process) mode.

Usage:
################################################################################
cat << EOF > test_fuzzer.cc
#include <stddef.h>
#include <stdint.h>
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  if (size > 0 && data[0] == 'H')
    if (size > 1 && data[1] == 'I')
       if (size > 2 && data[2] == '!')
       __builtin_trap();
  return 0;
}
EOF
# Build your target with -fsanitize-coverage=trace-pc-guard using fresh clang.
clang -g -fsanitize-coverage=trace-pc-guard test_fuzzer.cc -c
# Build afl-llvm-rt.o.c from the AFL distribution.
clang -c -w $AFL_HOME/llvm_mode/afl-llvm-rt.o.c
# Build this file, link it with afl-llvm-rt.o.o and the target code.
clang++ afl_driver.cpp test_fuzzer.o afl-llvm-rt.o.o
# Run AFL:
rm -rf IN OUT; mkdir IN OUT; echo z > IN/z;
$AFL_HOME/afl-fuzz -i IN -o OUT ./a.out
################################################################################
Environment Variables:
There are a few environment variables that can be set to use features that
afl-fuzz doesn't have.

AFL_DRIVER_STDERR_DUPLICATE_FILENAME: Setting this *appends* stderr to the file
specified. If the file does not exist, it is created. This is useful for getting
stack traces (when using ASAN for example) or original error messages on hard to
reproduce bugs.

AFL_DRIVER_EXTRA_STATS_FILENAME: Setting this causes afl_driver to write extra
statistics to the file specified. Currently these are peak_rss_mb
(the peak amount of virtual memory used in MB) and slowest_unit_time_secs. If
the file does not exist it is created. If the file does exist then
afl_driver assumes it was restarted by afl-fuzz and will try to read old
statistics from the file. If that fails then the process will quit.

*/
#include <assert.h>
#include <errno.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <unistd.h>
#include <limits.h>

#include <fstream>
#include <iostream>
#include <vector>

// Platform detection. Copied from FuzzerInternal.h
#ifdef __linux__
#define LIBFUZZER_LINUX 1
#define LIBFUZZER_APPLE 0
#define LIBFUZZER_NETBSD 0
#define LIBFUZZER_FREEBSD 0
#define LIBFUZZER_OPENBSD 0
#elif __APPLE__
#define LIBFUZZER_LINUX 0
#define LIBFUZZER_APPLE 1
#define LIBFUZZER_NETBSD 0
#define LIBFUZZER_FREEBSD 0
#define LIBFUZZER_OPENBSD 0
#elif __NetBSD__
#define LIBFUZZER_LINUX 0
#define LIBFUZZER_APPLE 0
#define LIBFUZZER_NETBSD 1
#define LIBFUZZER_FREEBSD 0
#define LIBFUZZER_OPENBSD 0
#elif __FreeBSD__
#define LIBFUZZER_LINUX 0
#define LIBFUZZER_APPLE 0
#define LIBFUZZER_NETBSD 0
#define LIBFUZZER_FREEBSD 1
#define LIBFUZZER_OPENBSD 0
#elif __OpenBSD__
#define LIBFUZZER_LINUX 0
#define LIBFUZZER_APPLE 0
#define LIBFUZZER_NETBSD 0
#define LIBFUZZER_FREEBSD 0
#define LIBFUZZER_OPENBSD 1
#else
#error "Support for your platform has not been implemented"
#endif

// Used to avoid repeating error checking boilerplate. If cond is false, a
// fatal error has occurred in the program. In this event print error_message
// to stderr and abort(). Otherwise do nothing. Note that setting
// AFL_DRIVER_STDERR_DUPLICATE_FILENAME may cause error_message to be appended
// to the file as well, if the error occurs after the duplication is performed.
#define CHECK_ERROR(cond, error_message)                                       \
  if (!(cond)) {                                                               \
    fprintf(stderr, "%s\n", (error_message));                                  \
    abort();                                                                   \
  }

// libFuzzer interface is thin, so we don't include any libFuzzer headers.
extern "C" {
int LLVMFuzzerTestOneInput(const uint8_t *Data, size_t Size);
__attribute__((weak)) int LLVMFuzzerInitialize(int *argc, char ***argv);
}

// Notify AFL about persistent mode.
static volatile char AFL_PERSISTENT[] = "##SIG_AFL_PERSISTENT##";
extern "C" int __afl_persistent_loop(unsigned int);
static volatile char suppress_warning2 = AFL_PERSISTENT[0];

// Notify AFL about deferred forkserver.
static volatile char AFL_DEFER_FORKSVR[] = "##SIG_AFL_DEFER_FORKSRV##";
extern "C" void  __afl_manual_init();
static volatile char suppress_warning1 = AFL_DEFER_FORKSVR[0];

// Input buffer.
static const size_t kMaxAflInputSize = 1 << 20;
static uint8_t AflInputBuf[kMaxAflInputSize];

// Variables we need for writing to the extra stats file.
static FILE *extra_stats_file = NULL;
static uint32_t previous_peak_rss = 0;
static time_t slowest_unit_time_secs = 0;
static const int kNumExtraStats = 2;
static const char *kExtraStatsFormatString = "peak_rss_mb            : %u\n"
                                             "slowest_unit_time_sec  : %u\n";

// Experimental feature to use afl_driver without AFL's deferred mode.
// Needs to run before __afl_auto_init.
__attribute__((constructor(0))) void __decide_deferred_forkserver(void) {
  if (getenv("AFL_DRIVER_DONT_DEFER")) {
    if (unsetenv("__AFL_DEFER_FORKSRV")) {
      perror("Failed to unset __AFL_DEFER_FORKSRV");
      abort();
    }
  }
}

// Copied from FuzzerUtil.cpp.
size_t GetPeakRSSMb() {
  struct rusage usage;
  if (getrusage(RUSAGE_SELF, &usage))
    return 0;
  if (LIBFUZZER_LINUX || LIBFUZZER_NETBSD || LIBFUZZER_FREEBSD ||
      LIBFUZZER_OPENBSD) {
    // ru_maxrss is in KiB
    return usage.ru_maxrss >> 10;
  } else if (LIBFUZZER_APPLE) {
    // ru_maxrss is in bytes
    return usage.ru_maxrss >> 20;
  }
  assert(0 && "GetPeakRSSMb() is not implemented for your platform");
  return 0;
}

// Based on SetSigaction in FuzzerUtil.cpp
static void SetSigaction(int signum,
                         void (*callback)(int, siginfo_t *, void *)) {
  struct sigaction sigact;
  memset(&sigact, 0, sizeof(sigact));
  sigact.sa_sigaction = callback;
  if (sigaction(signum, &sigact, 0)) {
    fprintf(stderr, "libFuzzer: sigaction failed with %d\n", errno);
    exit(1);
  }
}

// Write extra stats to the file specified by the user. If none is specified
// this function will never be called.
static void write_extra_stats() {
  uint32_t peak_rss = GetPeakRSSMb();

  if (peak_rss < previous_peak_rss)
    peak_rss = previous_peak_rss;

  int chars_printed = fprintf(extra_stats_file, kExtraStatsFormatString,
                              peak_rss, slowest_unit_time_secs);

  CHECK_ERROR(chars_printed != 0, "Failed to write extra_stats_file");

  CHECK_ERROR(fclose(extra_stats_file) == 0,
              "Failed to close extra_stats_file");
}

// Call write_extra_stats before we exit.
static void crash_handler(int, siginfo_t *, void *) {
  // Make sure we don't try calling write_extra_stats again if we crashed while
  // trying to call it.
  static bool first_crash = true;
  CHECK_ERROR(first_crash,
              "Crashed in crash signal handler. This is a bug in the fuzzer.");

  first_crash = false;
  write_extra_stats();
}

// If the user has specified an extra_stats_file through the environment
// variable AFL_DRIVER_EXTRA_STATS_FILENAME, then perform necessary set up
// to write stats to it on exit. If no file is specified, do nothing. Otherwise
// install signal and exit handlers to write to the file when the process exits.
// Then if the file doesn't exist create it and set extra stats to 0. But if it
// does exist then read the initial values of the extra stats from the file
// and check that the file is writable.
static void maybe_initialize_extra_stats() {
  // If AFL_DRIVER_EXTRA_STATS_FILENAME isn't set then we have nothing to do.
  char *extra_stats_filename = getenv("AFL_DRIVER_EXTRA_STATS_FILENAME");
  if (!extra_stats_filename)
    return;

  // Open the file and find the previous peak_rss_mb value.
  // This is necessary because the fuzzing process is restarted after N
  // iterations are completed. So we may need to get this value from a previous
  // process to be accurate.
  extra_stats_file = fopen(extra_stats_filename, "r");

  // If extra_stats_file already exists: read old stats from it.
  if (extra_stats_file) {
    int matches = fscanf(extra_stats_file, kExtraStatsFormatString,
                         &previous_peak_rss, &slowest_unit_time_secs);

    // Make sure we have read a real extra stats file and that we have used it
    // to set slowest_unit_time_secs and previous_peak_rss.
    CHECK_ERROR(matches == kNumExtraStats, "Extra stats file is corrupt");

    CHECK_ERROR(fclose(extra_stats_file) == 0, "Failed to close file");

    // Now open the file for writing.
    extra_stats_file = fopen(extra_stats_filename, "w");
    CHECK_ERROR(extra_stats_file,
                "Failed to open extra stats file for writing");
  } else {
    // Looks like this is the first time in a fuzzing job this is being called.
    extra_stats_file = fopen(extra_stats_filename, "w+");
    CHECK_ERROR(extra_stats_file, "failed to create extra stats file");
  }

  // Make sure that crash_handler gets called on any kind of fatal error.
  int crash_signals[] = {SIGSEGV, SIGBUS, SIGABRT, SIGILL, SIGFPE,  SIGINT,
                         SIGTERM};

  const size_t num_signals = sizeof(crash_signals) / sizeof(crash_signals[0]);

  for (size_t idx = 0; idx < num_signals; idx++)
    SetSigaction(crash_signals[idx], crash_handler);

  // Make sure it gets called on other kinds of exits.
  atexit(write_extra_stats);
}

// If the user asks us to duplicate stderr, then do it.
static void maybe_duplicate_stderr() {
  char* stderr_duplicate_filename =
      getenv("AFL_DRIVER_STDERR_DUPLICATE_FILENAME");

  if (!stderr_duplicate_filename)
    return;

  FILE* stderr_duplicate_stream =
      freopen(stderr_duplicate_filename, "a+", stderr);

  if (!stderr_duplicate_stream) {
    fprintf(
        stderr,
        "Failed to duplicate stderr to AFL_DRIVER_STDERR_DUPLICATE_FILENAME");
    abort();
  }
}

// Define LLVMFuzzerMutate to avoid link failures for targets that use it
// with libFuzzer's LLVMFuzzerCustomMutator.
extern "C" size_t LLVMFuzzerMutate(uint8_t *Data, size_t Size, size_t MaxSize) {
  assert(false && "LLVMFuzzerMutate should not be called from afl_driver");
  return 0;
}

// Execute any files provided as parameters.
int ExecuteFilesOnyByOne(int argc, char **argv) {
  for (int i = 1; i < argc; i++) {
    std::ifstream in(argv[i]);
    in.seekg(0, in.end);
    size_t length = in.tellg();
    in.seekg (0, in.beg);
    std::cout << "Reading " << length << " bytes from " << argv[i] << std::endl;
    // Allocate exactly length bytes so that we reliably catch buffer overflows.
    std::vector<char> bytes(length);
    in.read(bytes.data(), bytes.size());
    assert(in);
    LLVMFuzzerTestOneInput(reinterpret_cast<const uint8_t *>(bytes.data()),
                           bytes.size());
    std::cout << "Execution successful" << std::endl;
  }
  return 0;
}

static void set_iterations(int *N, const char *arg) {
  char *next_char;
  long NL = strtol(arg, &next_char, 10);
  if (NL < 1 || NL > INT_MAX || *next_char != '\0') {
    fprintf(stderr, "WARNING: iterations invalid `%s`\n",
        arg);
    ::exit(-1);
  }

  *N = static_cast<int>(NL);
}

int main(int argc, char **argv) {
  fprintf(stderr,
      "======================= INFO =========================\n"
      "This binary is built for AFL-fuzz.\n"
      "To run the target function on individual input(s) execute this:\n"
      "  %s < INPUT_FILE\n"
      "or\n"
      "  %s INPUT_FILE1 [INPUT_FILE2 ... ]\n"
      "To fuzz with afl-fuzz execute this:\n"
      "  afl-fuzz [afl-flags] %s [-N]\n"
      "afl-fuzz will run N iterations before "
      "re-spawning the process (default: 1000)\n"
      "======================================================\n",
          argv[0], argv[0], argv[0]);
  if (LLVMFuzzerInitialize)
    LLVMFuzzerInitialize(&argc, &argv);
  // Do any other expensive one-time initialization here.

  maybe_duplicate_stderr();
  maybe_initialize_extra_stats();

  if (!getenv("AFL_DRIVER_DONT_DEFER"))
    __afl_manual_init();

  int N = 1000;
  if (argc == 2 && argv[1][0] == '-')
    set_iterations(&N, argv[1] + 1);
  else if(argc == 2) {
    fprintf(stderr, "WARNING: using the deprecated call style `%s %d`\n",
        argv[0], N);
    set_iterations(&N, argv[1]);
  } else if (argc > 1)
    return ExecuteFilesOnyByOne(argc, argv);

  assert(N > 0);

  // Call LLVMFuzzerTestOneInput here so that coverage caused by initialization
  // on the first execution of LLVMFuzzerTestOneInput is ignored.
  uint8_t dummy_input[1] = {0};
  LLVMFuzzerTestOneInput(dummy_input, 1);

  time_t unit_time_secs;
  int num_runs = 0;
  while (__afl_persistent_loop(N)) {
    ssize_t n_read = read(0, AflInputBuf, kMaxAflInputSize);
    if (n_read > 0) {
      // Copy AflInputBuf into a separate buffer to let asan find buffer
      // overflows. Don't use unique_ptr/etc to avoid extra dependencies.
      uint8_t *copy = new uint8_t[n_read];
      memcpy(copy, AflInputBuf, n_read);

      struct timeval unit_start_time;
      CHECK_ERROR(gettimeofday(&unit_start_time, NULL) == 0,
                  "Calling gettimeofday failed");

      num_runs++;
      LLVMFuzzerTestOneInput(copy, n_read);

      struct timeval unit_stop_time;
      CHECK_ERROR(gettimeofday(&unit_stop_time, NULL) == 0,
                  "Calling gettimeofday failed");

      // Update slowest_unit_time_secs if we see a new max.
      unit_time_secs = unit_stop_time.tv_sec - unit_start_time.tv_sec;
      if (slowest_unit_time_secs < unit_time_secs)
        slowest_unit_time_secs = unit_time_secs;

      delete[] copy;
    }
  }
  fprintf(stderr, "%s: successfully executed %d input(s)\n", argv[0], num_runs);
}
