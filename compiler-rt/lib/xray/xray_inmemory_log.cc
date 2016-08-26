//===-- xray_inmemory_log.cc ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of XRay, a dynamic runtime instrumentation system.
//
// Implementation of a simple in-memory log of XRay events. This defines a
// logging function that's compatible with the XRay handler interface, and
// routines for exporting data to files.
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <fcntl.h>
#include <mutex>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <thread>
#include <unistd.h>
#include <x86intrin.h>

#include "sanitizer_common/sanitizer_libc.h"
#include "xray/xray_records.h"
#include "xray_flags.h"
#include "xray_interface_internal.h"

// __xray_InMemoryRawLog will use a thread-local aligned buffer capped to a
// certain size (32kb by default) and use it as if it were a circular buffer for
// events. We store simple fixed-sized entries in the log for external analysis.

extern "C" {
void __xray_InMemoryRawLog(int32_t FuncId, XRayEntryType Type);
}

namespace __xray {

std::mutex LogMutex;

static void retryingWriteAll(int Fd, char *Begin, char *End) {
  if (Begin == End)
    return;
  auto TotalBytes = std::distance(Begin, End);
  while (auto Written = write(Fd, Begin, TotalBytes)) {
    if (Written < 0) {
      if (errno == EINTR)
        continue; // Try again.
      Report("Failed to write; errno = %d", errno);
      return;
    }

    // FIXME: Figure out whether/how to assert properly.
    assert(static_cast<uint64_t>(Written) <= TotalBytes);
    TotalBytes -= Written;
    if (TotalBytes == 0)
      break;
    Begin += Written;
  }
}

static std::pair<ssize_t, bool> retryingReadSome(int Fd, char *Begin,
                                                 char *End) {
  auto BytesToRead = std::distance(Begin, End);
  ssize_t BytesRead;
  ssize_t TotalBytesRead = 0;
  while (BytesToRead && (BytesRead = read(Fd, Begin, BytesToRead))) {
    if (BytesRead == -1) {
      if (errno == EINTR)
        continue;
      Report("Read error; errno = %d", errno);
      return std::make_pair(TotalBytesRead, false);
    }

    TotalBytesRead += BytesRead;
    BytesToRead -= BytesRead;
    Begin += BytesRead;
  }
  return std::make_pair(TotalBytesRead, true);
}

static bool readValueFromFile(const char *Filename, long long *Value) {
  int Fd = open(Filename, O_RDONLY | O_CLOEXEC);
  if (Fd == -1)
    return false;
  static constexpr size_t BufSize = 256;
  char Line[BufSize] = {};
  ssize_t BytesRead;
  bool Success;
  std::tie(BytesRead, Success) = retryingReadSome(Fd, Line, Line + BufSize);
  if (!Success)
    return false;
  close(Fd);
  char *End = nullptr;
  long long Tmp = internal_simple_strtoll(Line, &End, 10);
  bool Result = false;
  if (Line[0] != '\0' && (*End == '\n' || *End == '\0')) {
    *Value = Tmp;
    Result = true;
  }
  return Result;
}

class ThreadExitFlusher {
  int Fd;
  XRayRecord *Start;
  size_t &Offset;

public:
  explicit ThreadExitFlusher(int Fd, XRayRecord *Start, size_t &Offset)
      : Fd(Fd), Start(Start), Offset(Offset) {}

  ~ThreadExitFlusher() {
    std::lock_guard<std::mutex> L(LogMutex);
    if (Fd > 0 && Start != nullptr) {
      retryingWriteAll(Fd, reinterpret_cast<char *>(Start),
                       reinterpret_cast<char *>(Start + Offset));
      // Because this thread's exit could be the last one trying to write to the
      // file and that we're not able to close out the file properly, we sync
      // instead and hope that the pending writes are flushed as the thread
      // exits.
      fsync(Fd);
    }
  }
};

} // namespace __xray

using namespace __xray;

void PrintToStdErr(const char *Buffer) { fprintf(stderr, "%s", Buffer); }

void __xray_InMemoryRawLog(int32_t FuncId, XRayEntryType Type) {
  using Buffer =
      std::aligned_storage<sizeof(XRayRecord), alignof(XRayRecord)>::type;
  static constexpr size_t BuffLen = 1024;
  thread_local static Buffer InMemoryBuffer[BuffLen] = {};
  thread_local static size_t Offset = 0;
  static int Fd = [] {
    // FIXME: Figure out how to make this less stderr-dependent.
    SetPrintfAndReportCallback(PrintToStdErr);
    // Open a temporary file once for the log.
    static char TmpFilename[256] = {};
    static char TmpWildcardPattern[] = "XXXXXX";
    auto E = internal_strncat(TmpFilename, flags()->xray_logfile_base,
                              sizeof(TmpFilename) - 10);
    if (static_cast<size_t>((E + 6) - TmpFilename) >
        (sizeof(TmpFilename) - 1)) {
      Report("XRay log file base too long: %s", flags()->xray_logfile_base);
      return -1;
    }
    internal_strncat(TmpFilename, TmpWildcardPattern,
                     sizeof(TmpWildcardPattern) - 1);
    int Fd = mkstemp(TmpFilename);
    if (Fd == -1) {
      Report("XRay: Failed opening temporary file '%s'; not logging events.",
             TmpFilename);
      return -1;
    }
    if (Verbosity())
      fprintf(stderr, "XRay: Log file in '%s'\n", TmpFilename);

    // Get the cycle frequency from SysFS on Linux.
    long long CPUFrequency = -1;
    if (readValueFromFile("/sys/devices/system/cpu/cpu0/tsc_freq_khz",
                          &CPUFrequency)) {
      CPUFrequency *= 1000;
    } else if (readValueFromFile(
                   "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq",
                   &CPUFrequency)) {
      CPUFrequency *= 1000;
    } else {
      Report("Unable to determine CPU frequency for TSC accounting.");
    }

    // Since we're here, we get to write the header. We set it up so that the
    // header will only be written once, at the start, and let the threads
    // logging do writes which just append.
    XRayFileHeader Header;
    Header.Version = 1;
    Header.Type = FileTypes::NAIVE_LOG;
    Header.CycleFrequency =
        CPUFrequency == -1 ? 0 : static_cast<uint64_t>(CPUFrequency);

    // FIXME: Actually check whether we have 'constant_tsc' and 'nonstop_tsc'
    // before setting the values in the header.
    Header.ConstantTSC = 1;
    Header.NonstopTSC = 1;
    retryingWriteAll(Fd, reinterpret_cast<char *>(&Header),
                     reinterpret_cast<char *>(&Header) + sizeof(Header));
    return Fd;
  }();
  if (Fd == -1)
    return;
  thread_local __xray::ThreadExitFlusher Flusher(
      Fd, reinterpret_cast<__xray::XRayRecord *>(InMemoryBuffer), Offset);
  thread_local pid_t TId = syscall(SYS_gettid);

  // First we get the useful data, and stuff it into the already aligned buffer
  // through a pointer offset.
  auto &R = reinterpret_cast<__xray::XRayRecord *>(InMemoryBuffer)[Offset];
  unsigned CPU;
  R.RecordType = RecordTypes::NORMAL;
  R.TSC = __rdtscp(&CPU);
  R.CPU = CPU;
  R.TId = TId;
  R.Type = Type;
  R.FuncId = FuncId;
  ++Offset;
  if (Offset == BuffLen) {
    std::lock_guard<std::mutex> L(LogMutex);
    auto RecordBuffer = reinterpret_cast<__xray::XRayRecord *>(InMemoryBuffer);
    retryingWriteAll(Fd, reinterpret_cast<char *>(RecordBuffer),
                     reinterpret_cast<char *>(RecordBuffer + Offset));
    Offset = 0;
  }
}

static auto Unused = [] {
  if (flags()->xray_naive_log)
    __xray_set_handler(__xray_InMemoryRawLog);
  return true;
}();
