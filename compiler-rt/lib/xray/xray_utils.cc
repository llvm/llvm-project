//===-- xray_utils.cc -------------------------------------------*- C++ -*-===//
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
//===----------------------------------------------------------------------===//
#include "xray_utils.h"

#include "sanitizer_common/sanitizer_allocator_internal.h"
#include "sanitizer_common/sanitizer_common.h"
#include "xray_allocator.h"
#include "xray_defs.h"
#include "xray_flags.h"
#include <cstdio>
#include <errno.h>
#include <fcntl.h>
#include <iterator>
#include <stdlib.h>
#include <sys/types.h>
#include <tuple>
#include <unistd.h>
#include <utility>

namespace __xray {

void printToStdErr(const char *Buffer) XRAY_NEVER_INSTRUMENT {
  fprintf(stderr, "%s", Buffer);
}

LogWriter::~LogWriter() {
  internal_close(Fd);
}

void LogWriter::WriteAll(const char *Begin, const char *End) XRAY_NEVER_INSTRUMENT {
  if (Begin == End)
    return;
  auto TotalBytes = std::distance(Begin, End);
  while (auto Written = write(Fd, Begin, TotalBytes)) {
    if (Written < 0) {
      if (errno == EINTR)
        continue; // Try again.
      Report("Failed to write; errno = %d\n", errno);
      return;
    }
    TotalBytes -= Written;
    if (TotalBytes == 0)
      break;
    Begin += Written;
  }
}

void LogWriter::Flush() XRAY_NEVER_INSTRUMENT {
  fsync(Fd);
}

LogWriter *LogWriter::Open() XRAY_NEVER_INSTRUMENT {
  // Open a temporary file once for the log.
  char TmpFilename[256] = {};
  char TmpWildcardPattern[] = "XXXXXX";
  auto **Argv = GetArgv();
  const char *Progname = !Argv ? "(unknown)" : Argv[0];
  const char *LastSlash = internal_strrchr(Progname, '/');

  if (LastSlash != nullptr)
    Progname = LastSlash + 1;

  int NeededLength = internal_snprintf(
      TmpFilename, sizeof(TmpFilename), "%s%s.%s",
      flags()->xray_logfile_base, Progname, TmpWildcardPattern);
  if (NeededLength > int(sizeof(TmpFilename))) {
    Report("XRay log file name too long (%d): %s\n", NeededLength, TmpFilename);
    return nullptr;
  }
  int Fd = mkstemp(TmpFilename);
  if (Fd == -1) {
    Report("XRay: Failed opening temporary file '%s'; not logging events.\n",
           TmpFilename);
    return nullptr;
  }
  if (Verbosity())
    Report("XRay: Log file in '%s'\n", TmpFilename);

  LogWriter *LW = allocate<LogWriter>();
  new (LW) LogWriter(Fd);
  return LW;
}

void LogWriter::Close(LogWriter *LW) {
  LW->~LogWriter();
  deallocate(LW);
}

} // namespace __xray
