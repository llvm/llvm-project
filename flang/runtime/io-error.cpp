//===-- runtime/io-error.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "io-error.h"
#include "config.h"
#include "tools.h"
#include "flang/Runtime/magic-numbers.h"
#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstring>

namespace Fortran::runtime::io {
RT_OFFLOAD_API_GROUP_BEGIN

void IoErrorHandler::SignalError(int iostatOrErrno, const char *msg, ...) {
  // Note that IOMSG= alone without IOSTAT=/END=/EOR=/ERR= does not suffice
  // for error recovery (see F'2018 subclause 12.11).
  switch (iostatOrErrno) {
  case IostatOk:
    return;
  case IostatEnd:
    if ((flags_ & (hasIoStat | hasEnd)) ||
        ((flags_ & hasErr) && (flags_ & hasRec))) {
      // EOF goes to ERR= when REC= is present
      if (ioStat_ == IostatOk || ioStat_ < IostatEnd) {
        ioStat_ = IostatEnd;
      }
      return;
    }
    break;
  case IostatEor:
    if (flags_ & (hasIoStat | hasEor)) {
      if (ioStat_ == IostatOk || ioStat_ < IostatEor) {
        ioStat_ = IostatEor; // least priority
      }
      return;
    }
    break;
  default:
    if (flags_ & (hasIoStat | hasErr)) {
      if (ioStat_ <= 0) {
        ioStat_ = iostatOrErrno; // priority over END=/EOR=
        if (msg && (flags_ & hasIoMsg)) {
#if !defined(RT_DEVICE_COMPILATION)
          char buffer[256];
          va_list ap;
          va_start(ap, msg);
          std::vsnprintf(buffer, sizeof buffer, msg, ap);
          va_end(ap);
#else
          const char *buffer = "not implemented yet: IOSTAT with varargs";
#endif
          ioMsg_ = SaveDefaultCharacter(
              buffer, Fortran::runtime::strlen(buffer) + 1, *this);
        }
      }
      return;
    }
    break;
  }
  // I/O error not caught!
  if (msg) {
#if !defined(RT_DEVICE_COMPILATION)
    va_list ap;
    va_start(ap, msg);
    CrashArgs(msg, ap);
    va_end(ap);
#else
    Crash("not implemented yet: IOSTAT with varargs");
#endif
  } else if (const char *errstr{IostatErrorString(iostatOrErrno)}) {
    Crash(errstr);
  } else {
#if !defined(RT_DEVICE_COMPILATION)
    Crash("I/O error (errno=%d): %s", iostatOrErrno,
        std::strerror(iostatOrErrno));
#else
    Crash("I/O error (errno=%d)", iostatOrErrno);
#endif
  }
}

void IoErrorHandler::SignalError(int iostatOrErrno) {
  SignalError(iostatOrErrno, nullptr);
}

void IoErrorHandler::Forward(
    int ioStatOrErrno, const char *msg, std::size_t length) {
  if (ioStatOrErrno != IostatOk) {
    if (msg) {
      SignalError(ioStatOrErrno, "%.*s", static_cast<int>(length), msg);
    } else {
      SignalError(ioStatOrErrno);
    }
  }
}

void IoErrorHandler::SignalEnd() { SignalError(IostatEnd); }

void IoErrorHandler::SignalEor() { SignalError(IostatEor); }

void IoErrorHandler::SignalPendingError() {
  int error{pendingError_};
  pendingError_ = IostatOk;
  SignalError(error);
}

void IoErrorHandler::SignalErrno() { SignalError(errno); }

bool IoErrorHandler::GetIoMsg(char *buffer, std::size_t bufferLength) {
  const char *msg{ioMsg_.get()};
  if (!msg) {
    msg = IostatErrorString(ioStat_ == IostatOk ? pendingError_ : ioStat_);
  }
  if (msg) {
    ToFortranDefaultCharacter(buffer, bufferLength, msg);
    return true;
  }

  // Following code is taken from llvm/lib/Support/Errno.cpp
  // in LLVM v9.0.1 with inadequate modification for Fortran,
  // since rectified.
  bool ok{false};
#if defined(RT_DEVICE_COMPILATION)
  // strerror_r is not available on device.
  msg = "errno description is not available on device";
#elif HAVE_STRERROR_R
  // strerror_r is thread-safe.
#if defined(__GLIBC__) && defined(_GNU_SOURCE)
  // glibc defines its own incompatible version of strerror_r
  // which may not use the buffer supplied.
  msg = ::strerror_r(ioStat_, buffer, bufferLength);
#else
  ok = ::strerror_r(ioStat_, buffer, bufferLength) == 0;
#endif
#elif HAVE_DECL_STRERROR_S // "Windows Secure API"
  ok = ::strerror_s(buffer, bufferLength, ioStat_) == 0;
#else
  // Copy the thread un-safe result of strerror into
  // the buffer as fast as possible to minimize impact
  // of collision of strerror in multiple threads.
  msg = strerror(ioStat_);
#endif
  if (msg) {
    ToFortranDefaultCharacter(buffer, bufferLength, msg);
    return true;
  } else if (ok) {
    std::size_t copied{Fortran::runtime::strlen(buffer)};
    if (copied < bufferLength) {
      std::memset(buffer + copied, ' ', bufferLength - copied);
    }
    return true;
  } else {
    return false;
  }
}

RT_OFFLOAD_API_GROUP_END
} // namespace Fortran::runtime::io
