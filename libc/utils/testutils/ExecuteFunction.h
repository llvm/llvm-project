//===-- ExecuteFunction.h ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_TESTUTILS_EXECUTEFUNCTION_H
#define LLVM_LIBC_UTILS_TESTUTILS_EXECUTEFUNCTION_H

#include <stdint.h>

namespace __llvm_libc {
namespace testutils {

class FunctionCaller {
public:
  virtual ~FunctionCaller() {}
  virtual void operator()() = 0;
};

struct ProcessStatus {
  int PlatformDefined;
  const char *failure = nullptr;

  static constexpr uintptr_t timeout = -1L;

  static ProcessStatus Error(const char *error) { return {0, error}; }
  static ProcessStatus TimedOut() {
    return {0, reinterpret_cast<const char *>(timeout)};
  }

  bool timedOut() const {
    return failure == reinterpret_cast<const char *>(timeout);
  }
  const char *getError() const { return timedOut() ? nullptr : failure; }
  bool exitedNormally() const;
  int getExitCode() const;
  int getFatalSignal() const;
};

ProcessStatus invokeInSubprocess(FunctionCaller *Func, unsigned TimeoutMS = -1);

const char *signalAsString(int Signum);

} // namespace testutils
} // namespace __llvm_libc

#endif // LLVM_LIBC_UTILS_TESTUTILS_EXECUTEFUNCTION_H
