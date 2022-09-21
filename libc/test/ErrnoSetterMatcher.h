//===-- ErrnoSetterMatcher.h ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_TEST_ERRNOSETTERMATCHER_H
#define LLVM_LIBC_TEST_ERRNOSETTERMATCHER_H

#include "utils/UnitTest/Test.h"

#include <errno.h>

namespace __llvm_libc {
namespace testing {

namespace internal {

extern "C" char *strerror(int);

template <typename T> class ErrnoSetterMatcher : public Matcher<T> {
  T ExpectedReturn;
  T ActualReturn;
  int ExpectedErrno;
  int ActualErrno;

public:
  ErrnoSetterMatcher(T ExpectedReturn, int ExpectedErrno)
      : ExpectedReturn(ExpectedReturn), ExpectedErrno(ExpectedErrno) {}

  void explainError(testutils::StreamWrapper &OS) override {
    if (ActualReturn != ExpectedReturn)
      OS << "Expected return to be " << ExpectedReturn << " but got "
         << ActualReturn << ".\nExpecte errno to be " << strerror(ExpectedErrno)
         << " but got " << strerror(ActualErrno) << ".\n";
    else
      OS << "Correct value " << ExpectedReturn
         << " was returned\nBut errno was unexpectely set to "
         << strerror(ActualErrno) << ".\n";
  }

  bool match(T Got) {
    ActualReturn = Got;
    ActualErrno = errno;
    errno = 0;
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
