//===---------------------- ExecuteFunction.h -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_TESTUTILS_EXECUTEFUNCTION_H
#define LLVM_LIBC_UTILS_TESTUTILS_EXECUTEFUNCTION_H

namespace __llvm_libc {
namespace testutils {

class FunctionCaller {
public:
  virtual ~FunctionCaller() {}
  virtual void operator()() = 0;
};

struct ProcessStatus {
  int PlatformDefined;

  bool exitedNormally();
  int getExitCode();
  int getFatalSignal();
};

ProcessStatus invokeInSubprocess(FunctionCaller *Func);

const char *signalAsString(int Signum);

} // namespace testutils
} // namespace __llvm_libc

#endif // LLVM_LIBC_UTILS_TESTUTILS_EXECUTEFUNCTION_H
