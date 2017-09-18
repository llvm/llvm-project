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
#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include "sanitizer_common/sanitizer_libc.h"
#include "xray/xray_records.h"
#include "xray_defs.h"
#include "xray_flags.h"
#include "xray_interface_internal.h"
#include "xray_tsc.h"
#include "xray_utils.h"

// __xray_InMemoryRawLog will use a thread-local aligned buffer capped to a
// certain size (32kb by default) and use it as if it were a circular buffer for
// events. We store simple fixed-sized entries in the log for external analysis.

extern "C" {
void __xray_InMemoryRawLog(int32_t FuncId,
                           XRayEntryType Type) XRAY_NEVER_INSTRUMENT;
}

namespace __xray {

__sanitizer::SpinMutex LogMutex;

class ThreadExitFlusher {
  int Fd;
  XRayRecord *Start;
  size_t &Offset;

public:
  explicit ThreadExitFlusher(int Fd, XRayRecord *Start,
                             size_t &Offset) XRAY_NEVER_INSTRUMENT
      : Fd(Fd),
        Start(Start),
        Offset(Offset) {}

  ~ThreadExitFlusher() XRAY_NEVER_INSTRUMENT {
    __sanitizer::SpinMutexLock L(&LogMutex);
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

static int __xray_OpenLogFile() XRAY_NEVER_INSTRUMENT {
  int F = getLogFD();
  if (F == -1)
    return -1;

  // Test for required CPU features and cache the cycle frequency
  static bool TSCSupported = probeRequiredCPUFeatures();
  static uint64_t CycleFrequency = TSCSupported ? getTSCFrequency()
                                   : __xray::NanosecondsPerSecond;

  // Since we're here, we get to write the header. We set it up so that the
  // header will only be written once, at the start, and let the threads
  // logging do writes which just append.
  XRayFileHeader Header;
  Header.Version = 2;  // Version 2 includes tail exit records.
  Header.Type = FileTypes::NAIVE_LOG;
  Header.CycleFrequency = CycleFrequency;

  // FIXME: Actually check whether we have 'constant_tsc' and 'nonstop_tsc'
  // before setting the values in the header.
  Header.ConstantTSC = 1;
  Header.NonstopTSC = 1;
  retryingWriteAll(F, reinterpret_cast<char *>(&Header),
                   reinterpret_cast<char *>(&Header) + sizeof(Header));
  return F;
}

template <class RDTSC>
void __xray_InMemoryRawLog(int32_t FuncId, XRayEntryType Type,
                           RDTSC ReadTSC) XRAY_NEVER_INSTRUMENT {
  using Buffer =
      std::aligned_storage<sizeof(XRayRecord), alignof(XRayRecord)>::type;
  static constexpr size_t BuffLen = 1024;
  thread_local static Buffer InMemoryBuffer[BuffLen] = {};
  thread_local static size_t Offset = 0;
  static int Fd = __xray_OpenLogFile();
  if (Fd == -1)
    return;
  thread_local __xray::ThreadExitFlusher Flusher(
      Fd, reinterpret_cast<__xray::XRayRecord *>(InMemoryBuffer), Offset);
  thread_local pid_t TId = syscall(SYS_gettid);

  // Use a simple recursion guard, to handle cases where we're already logging
  // and for one reason or another, this function gets called again in the same
  // thread.
  thread_local volatile bool RecusionGuard = false;
  if (RecusionGuard) return;
  RecusionGuard = true;

  // First we get the useful data, and stuff it into the already aligned buffer
  // through a pointer offset.
  auto &R = reinterpret_cast<__xray::XRayRecord *>(InMemoryBuffer)[Offset];
  R.RecordType = RecordTypes::NORMAL;
  R.TSC = ReadTSC(R.CPU);
  R.TId = TId;
  R.Type = Type;
  R.FuncId = FuncId;
  ++Offset;
  if (Offset == BuffLen) {
    __sanitizer::SpinMutexLock L(&LogMutex);
    auto RecordBuffer = reinterpret_cast<__xray::XRayRecord *>(InMemoryBuffer);
    retryingWriteAll(Fd, reinterpret_cast<char *>(RecordBuffer),
                     reinterpret_cast<char *>(RecordBuffer + Offset));
    Offset = 0;
  }

  RecusionGuard = false;
}

void __xray_InMemoryRawLogRealTSC(int32_t FuncId,
                                  XRayEntryType Type) XRAY_NEVER_INSTRUMENT {
  __xray_InMemoryRawLog(FuncId, Type, __xray::readTSC);
}

void __xray_InMemoryEmulateTSC(int32_t FuncId,
                               XRayEntryType Type) XRAY_NEVER_INSTRUMENT {
  __xray_InMemoryRawLog(FuncId, Type, [](uint8_t &CPU) XRAY_NEVER_INSTRUMENT {
    timespec TS;
    int result = clock_gettime(CLOCK_REALTIME, &TS);
    if (result != 0) {
      Report("clock_gettimg(2) return %d, errno=%d.", result, int(errno));
      TS = {0, 0};
    }
    CPU = 0;
    return TS.tv_sec * __xray::NanosecondsPerSecond + TS.tv_nsec;
  });
}

static auto UNUSED Unused = [] {
  auto UseRealTSC = probeRequiredCPUFeatures();
  if (!UseRealTSC)
    Report("WARNING: Required CPU features missing for XRay instrumentation, "
           "using emulation instead.\n");
  if (flags()->xray_naive_log)
    __xray_set_handler(UseRealTSC ? __xray_InMemoryRawLogRealTSC
                                  : __xray_InMemoryEmulateTSC);
  return true;
}();
