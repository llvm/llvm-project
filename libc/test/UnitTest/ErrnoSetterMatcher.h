//===-- ErrnoSetterMatcher.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_ERRNOSETTERMATCHER_H
#define LLVM_LIBC_TEST_ERRNOSETTERMATCHER_H

#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/fpbits_str.h"
#include "src/__support/StringUtil/error_to_string.h"
#include "src/__support/macros/properties/architectures.h"
#include "src/errno/libc_errno.h"
#include "test/UnitTest/Test.h"

namespace __llvm_libc {
namespace testing {

namespace internal {

template <typename T> class ErrnoSetterMatcher : public Matcher<T> {
  T ExpectedReturn;
  T ActualReturn;
  int ExpectedErrno;
  int ActualErrno;

  // Even though this is a errno matcher primarily, it has to cater to platforms
  // which do not have an errno. This predicate checks if errno matching is to
  // be skipped.
  static constexpr bool ignore_errno() {
#ifdef LIBC_TARGET_ARCH_IS_GPU
    return true;
#else
    return false;
#endif
  }

public:
  ErrnoSetterMatcher(T ExpectedReturn, int ExpectedErrno)
      : ExpectedReturn(ExpectedReturn), ExpectedErrno(ExpectedErrno) {}

  void explainError() override {
    if (ActualReturn != ExpectedReturn) {
      if constexpr (cpp::is_floating_point_v<T>) {
        tlog << "Expected return value to be: "
             << str(fputil::FPBits<T>(ExpectedReturn)) << '\n';
        tlog << "                    But got: "
             << str(fputil::FPBits<T>(ActualReturn)) << '\n';
      } else {
        tlog << "Expected return value to be " << ExpectedReturn << " but got "
             << ActualReturn << ".\n";
      }
    }

    if constexpr (!ignore_errno()) {
      if (ActualErrno != ExpectedErrno) {
        tlog << "Expected errno to be \"" << get_error_string(ExpectedErrno)
             << "\" but got \"" << get_error_string(ActualErrno) << "\".\n";
      }
    }
  }

  bool match(T Got) {
    ActualReturn = Got;
    ActualErrno = libc_errno;
    libc_errno = 0;
    if constexpr (ignore_errno())
      return Got == ExpectedReturn;
    else
      return Got == ExpectedReturn && ActualErrno == ExpectedErrno;
  }
};

} // namespace internal

namespace ErrnoSetterMatcher {

template <typename RetT = int>
static internal::ErrnoSetterMatcher<RetT> Succeeds(RetT ExpectedReturn = 0,
                                                   int ExpectedErrno = 0) {
  return {ExpectedReturn, ExpectedErrno};
}

template <typename RetT = int>
static internal::ErrnoSetterMatcher<RetT> Fails(int ExpectedErrno,
                                                RetT ExpectedReturn = -1) {
  return {ExpectedReturn, ExpectedErrno};
}

} // namespace ErrnoSetterMatcher

} // namespace testing
} // namespace __llvm_libc

#endif // LLVM_LIBC_TEST_ERRNOSETTERMATCHER_H
