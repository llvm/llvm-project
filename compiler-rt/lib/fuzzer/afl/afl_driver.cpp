//===- afl_driver.cpp - a glue between AFL and libFuzzer --------*- C++ -* ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
AFL_DRIVER_STDERR_DUPLICATE_FILENAME: Setting this environment variable
*appends* stderr to the file specified. If the file does not exist, it is
created. This is useful for getting stack traces (when using ASAN for example)
or original error messages on hard to reproduce bugs.

*/
#include <assert.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

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
    std::ifstream in(argv[i], std::ios::binary);
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

  if (!getenv("AFL_DRIVER_DONT_DEFER"))
    __afl_manual_init();

  int N = 1000;
  if (argc == 2 && argv[1][0] == '-')
      N = atoi(argv[1] + 1);
  else if(argc == 2 && (N = atoi(argv[1])) > 0)
      fprintf(stderr, "WARNING: using the deprecated call style `%s %d`\n",
              argv[0], N);
  else if (argc > 1)
    return ExecuteFilesOnyByOne(argc, argv);

  assert(N > 0);

  // Call LLVMFuzzerTestOneInput here so that coverage caused by initialization
  // on the first execution of LLVMFuzzerTestOneInput is ignored.
  uint8_t dummy_input[1] = {0};
  LLVMFuzzerTestOneInput(dummy_input, 1);

  int num_runs = 0;
  while (__afl_persistent_loop(N)) {
    ssize_t n_read = read(0, AflInputBuf, kMaxAflInputSize);
    if (n_read > 0) {
      // Copy AflInputBuf into a separate buffer to let asan find buffer
      // overflows. Don't use unique_ptr/etc to avoid extra dependencies.
      uint8_t *copy = new uint8_t[n_read];
      memcpy(copy, AflInputBuf, n_read);
      num_runs++;
      LLVMFuzzerTestOneInput(copy, n_read);
      delete[] copy;
    }
  }
  fprintf(stderr, "%s: successfully executed %d input(s)\n", argv[0], num_runs);
}
