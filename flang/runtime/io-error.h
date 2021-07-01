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

#include "iostat.h"
#include "memory.h"
#include "terminator.h"
#include <cinttypes>

namespace Fortran::runtime::io {

// See 12.11 in Fortran 2018
class IoErrorHandler : public Terminator {
public:
  using Terminator::Terminator;
  explicit IoErrorHandler(const Terminator &that) : Terminator{that} {}
  void HasIoStat() { flags_ |= hasIoStat; }
  void HasErrLabel() { flags_ |= hasErr; }
  void HasEndLabel() { flags_ |= hasEnd; }
  void HasEorLabel() { flags_ |= hasEor; }
  void HasIoMsg() { flags_ |= hasIoMsg; }
  void HandleAnything() {
    flags_ = hasIoStat | hasErr | hasEnd | hasEor | hasIoMsg;
  }

  bool InError() const { return ioStat_ != IostatOk; }

  void SignalError(int iostatOrErrno, const char *msg, ...);
  void SignalError(int iostatOrErrno);
  template <typename... X> void SignalError(const char *msg, X &&...xs) {
    SignalError(IostatGenericError, msg, std::forward<X>(xs)...);
  }

  void Forward(int iostatOrErrno, const char *, std::size_t);

  void SignalErrno(); // SignalError(errno)
  void SignalEnd(); // input only; EOF on internal write is an error
  void SignalEor(); // non-advancing input only; EOR on write is an error

  int GetIoStat() const { return ioStat_; }
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
};

} // namespace Fortran::runtime::io
#endif // FORTRAN_RUNTIME_IO_ERROR_H_
