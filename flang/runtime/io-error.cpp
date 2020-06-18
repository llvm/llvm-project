//===-- runtime/io-error.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "io-error.h"
#include "config.h"
#include "magic-numbers.h"
#include "tools.h"
#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstring>

namespace Fortran::runtime::io {

void IoErrorHandler::Begin(const char *sourceFileName, int sourceLine) {
  flags_ = 0;
  ioStat_ = 0;
  ioMsg_.reset();
  SetLocation(sourceFileName, sourceLine);
}

void IoErrorHandler::SignalError(int iostatOrErrno, const char *msg, ...) {
  if (iostatOrErrno == IostatEnd && (flags_ & hasEnd)) {
    if (!ioStat_ || ioStat_ < IostatEnd) {
      ioStat_ = IostatEnd;
    }
  } else if (iostatOrErrno == IostatEor && (flags_ & hasEor)) {
    if (!ioStat_ || ioStat_ < IostatEor) {
      ioStat_ = IostatEor; // least priority
    }
  } else if (iostatOrErrno != IostatOk) {
    if (flags_ & (hasIoStat | hasErr)) {
      if (ioStat_ <= 0) {
        ioStat_ = iostatOrErrno; // priority over END=/EOR=
        if (msg && (flags_ & hasIoMsg)) {
          char buffer[256];
          va_list ap;
          va_start(ap, msg);
          std::vsnprintf(buffer, sizeof buffer, msg, ap);
          ioMsg_ = SaveDefaultCharacter(buffer, std::strlen(buffer) + 1, *this);
        }
      }
    } else if (msg) {
      va_list ap;
      va_start(ap, msg);
      CrashArgs(msg, ap);
    } else if (const char *errstr{IostatErrorString(iostatOrErrno)}) {
      Crash(errstr);
    } else {
      Crash("I/O error (errno=%d): %s", iostatOrErrno,
          std::strerror(iostatOrErrno));
    }
  }
}

void IoErrorHandler::SignalError(int iostatOrErrno) {
  SignalError(iostatOrErrno, nullptr);
}

void IoErrorHandler::SignalErrno() { SignalError(errno); }

void IoErrorHandler::SignalEnd() { SignalError(IostatEnd); }

void IoErrorHandler::SignalEor() { SignalError(IostatEor); }

bool IoErrorHandler::GetIoMsg(char *buffer, std::size_t bufferLength) {
  const char *msg{ioMsg_.get()};
  if (!msg) {
    msg = IostatErrorString(ioStat_);
  }
  if (msg) {
    ToFortranDefaultCharacter(buffer, bufferLength, msg);
    return true;
  }

  char *newBuf;
  // Following code is taken from llvm/lib/Support/Errno.cpp
  // in LLVM v9.0.1
#if HAVE_STRERROR_R
  // strerror_r is thread-safe.
#if defined(__GLIBC__) && defined(_GNU_SOURCE)
  // glibc defines its own incompatible version of strerror_r
  // which may not use the buffer supplied.
  newBuf = ::strerror_r(ioStat_, buffer, bufferLength);
#else
  return ::strerror_r(ioStat_, buffer, bufferLength) == 0;
#endif
#elif HAVE_DECL_STRERROR_S // "Windows Secure API"
  return ::strerror_s(buffer, bufferLength, ioStat_) == 0;
#elif HAVE_STRERROR
  // Copy the thread un-safe result of strerror into
  // the buffer as fast as possible to minimize impact
  // of collision of strerror in multiple threads.
  newBuf = strerror(ioStat_);
#else
  // Strange that this system doesn't even have strerror
  return false;
#endif
  ::strncpy(buffer, newBuf, bufferLength - 1);
  buffer[bufferLength - 1] = '\n';
  return true;
}
} // namespace Fortran::runtime::io
