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
  int platform_defined;
  const char *failure = nullptr;

  static constexpr uintptr_t TIMEOUT = -1L;

  static ProcessStatus error(const char *error) { return {0, error}; }
  static ProcessStatus timed_out_ps() {
    return {0, reinterpret_cast<const char *>(TIMEOUT)};
  }

  bool timed_out() const {
    return failure == reinterpret_cast<const char *>(TIMEOUT);
  }
  const char *get_error() const { return timed_out() ? nullptr : failure; }
  bool exited_normally() const;
  int get_exit_code() const;
  int get_fatal_signal() const;
};

ProcessStatus invoke_in_subprocess(FunctionCaller *Func,
                                   unsigned TimeoutMS = -1);

const char *signal_as_string(int Signum);

} // namespace testutils
} // namespace __llvm_libc

#endif // LLVM_LIBC_UTILS_TESTUTILS_EXECUTEFUNCTION_H
