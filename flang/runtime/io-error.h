//===-- runtime/io-error.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Distinguishes I/O error conditions; fatal ones lead to termination,
// and those that the user program has chosen to handle are recorded
// so that the highest-priority one can be returned as IOSTAT=.
// IOSTAT error codes are raw errno values augmented with values for
// Fortran-specific errors.

#ifndef FORTRAN_RUNTIME_IO_ERROR_H_
#define FORTRAN_RUNTIME_IO_ERROR_H_

#include "terminator.h"
#include "flang/Runtime/iostat.h"
#include "flang/Runtime/memory.h"
#include <cinttypes>

namespace Fortran::runtime::io {

// See 12.11 in Fortran 2018
class IoErrorHandler : public Terminator {
public:
  using Terminator::Terminator;
  explicit RT_API_ATTRS IoErrorHandler(const Terminator &that)
      : Terminator{that} {}
  RT_API_ATTRS void HasIoStat() { flags_ |= hasIoStat; }
  RT_API_ATTRS void HasErrLabel() { flags_ |= hasErr; }
  RT_API_ATTRS void HasEndLabel() { flags_ |= hasEnd; }
  RT_API_ATTRS void HasEorLabel() { flags_ |= hasEor; }
  RT_API_ATTRS void HasIoMsg() { flags_ |= hasIoMsg; }

  RT_API_ATTRS bool InError() const {
    return ioStat_ != IostatOk || pendingError_ != IostatOk;
  }

  // For I/O statements that detect fatal errors in their
  // Begin...() API routines before it is known whether they
  // have error handling control list items.  Such statements
  // have an ErroneousIoStatementState with a pending error.
  RT_API_ATTRS void SetPendingError(int iostat) { pendingError_ = iostat; }

  RT_API_ATTRS void SignalError(int iostatOrErrno, const char *msg, ...);
  RT_API_ATTRS void SignalError(int iostatOrErrno);
  template <typename... X>
  RT_API_ATTRS void SignalError(const char *msg, X &&...xs) {
    SignalError(IostatGenericError, msg, std::forward<X>(xs)...);
  }

  RT_API_ATTRS void Forward(int iostatOrErrno, const char *, std::size_t);

  void SignalErrno(); // SignalError(errno)
  RT_API_ATTRS void
  SignalEnd(); // input only; EOF on internal write is an error
  RT_API_ATTRS void
  SignalEor(); // non-advancing input only; EOR on write is an error
  RT_API_ATTRS void SignalPendingError();

  RT_API_ATTRS int GetIoStat() const { return ioStat_; }
  bool GetIoMsg(char *, std::size_t);

private:
  enum Flag : std::uint8_t {
    hasIoStat = 1, // IOSTAT=
    hasErr = 2, // ERR=
    hasEnd = 4, // END=
    hasEor = 8, // EOR=
    hasIoMsg = 16, // IOMSG=
  };
  std::uint8_t flags_{0};
  int ioStat_{IostatOk};
  OwningPtr<char> ioMsg_;
  int pendingError_{IostatOk};
};

} // namespace Fortran::runtime::io
#endif // FORTRAN_RUNTIME_IO_ERROR_H_
