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

#include "sanitizer_common/sanitizer_common.h"
#include "xray_defs.h"
#include "xray_flags.h"
#include <stdlib.h>
#include <cstdio>
#include <errno.h>
#include <fcntl.h>
#include <iterator>
#include <sys/types.h>
#include <tuple>
#include <unistd.h>
#include <utility>

namespace __xray {

void printToStdErr(const char *Buffer) XRAY_NEVER_INSTRUMENT {
  fprintf(stderr, "%s", Buffer);
}

void retryingWriteAll(int Fd, char *Begin, char *End) XRAY_NEVER_INSTRUMENT {
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

std::pair<ssize_t, bool> retryingReadSome(int Fd, char *Begin,
                                          char *End) XRAY_NEVER_INSTRUMENT {
  auto BytesToRead = std::distance(Begin, End);
  ssize_t BytesRead;
  ssize_t TotalBytesRead = 0;
  while (BytesToRead && (BytesRead = read(Fd, Begin, BytesToRead))) {
    if (BytesRead == -1) {
      if (errno == EINTR)
        continue;
      Report("Read error; errno = %d\n", errno);
      return std::make_pair(TotalBytesRead, false);
    }

    TotalBytesRead += BytesRead;
    BytesToRead -= BytesRead;
    Begin += BytesRead;
  }
  return std::make_pair(TotalBytesRead, true);
}

bool readValueFromFile(const char *Filename,
                       long long *Value) XRAY_NEVER_INSTRUMENT {
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

int getLogFD() XRAY_NEVER_INSTRUMENT {
  // Open a temporary file once for the log.
  static char TmpFilename[256] = {};
  static char TmpWildcardPattern[] = "XXXXXX";
  auto Argv = GetArgv();
  const char *Progname = Argv[0] == nullptr ? "(unknown)" : Argv[0];
  const char *LastSlash = internal_strrchr(Progname, '/');

  if (LastSlash != nullptr)
    Progname = LastSlash + 1;

  const int HalfLength = sizeof(TmpFilename) / 2 - sizeof(TmpWildcardPattern);
  int NeededLength = internal_snprintf(
      TmpFilename, sizeof(TmpFilename), "%.*s%.*s.%s", HalfLength,
      flags()->xray_logfile_base, HalfLength, Progname, TmpWildcardPattern);
  if (NeededLength > int(sizeof(TmpFilename))) {
    Report("XRay log file name too long (%d): %s\n", NeededLength, TmpFilename);
    return -1;
  }
  int Fd = mkstemp(TmpFilename);
  if (Fd == -1) {
    Report("XRay: Failed opening temporary file '%s'; not logging events.\n",
           TmpFilename);
    return -1;
  }
  Report("XRay: Log file in '%s'\n", TmpFilename);

  return Fd;
}

} // namespace __xray
