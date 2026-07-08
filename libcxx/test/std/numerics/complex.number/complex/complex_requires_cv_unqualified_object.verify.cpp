//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: FROZEN-CXX03-HEADERS-FIXME

// <complex>

// template<class T>
// class complex
//
// T shall be a cv-unqualified object type (LWG3133)

#include <complex>

struct NotDefaultConstructible {
  NotDefaultConstructible() = delete;
};

struct NotCopyConstructible {
  NotCopyConstructible()                            = default;
  NotCopyConstructible(const NotCopyConstructible&) = delete;
};

struct NotCopyAssignable {
  NotCopyAssignable()                                    = default;
  NotCopyAssignable(const NotCopyAssignable&)            = default;
  NotCopyAssignable& operator=(const NotCopyAssignable&) = delete;
};

struct NotDestructible {
  NotDestructible()                                  = default;
  NotDestructible(const NotDestructible&)            = default;
  NotDestructible& operator=(const NotDestructible&) = default;

private:
  ~NotDestructible() = default;
};

void test() {
  // expected-error-re@complex:* {{static assertion failed{{.*}}std::complex<T> requires T to be a cv-unqualified object type}}
  (void)sizeof(std::complex<const double>);
  // expected-error-re@complex:* {{static assertion failed{{.*}}std::complex<T> requires T to be a cv-unqualified object type}}
  (void)sizeof(std::complex<volatile int>);
  // expected-error-re@complex:* {{static assertion failed{{.*}}std::complex<T> requires T to be a cv-unqualified object type}}
  (void)sizeof(std::complex<const volatile float>);

  // expected-error-re@complex:* {{static assertion failed{{.*}}std::complex<T> requires T to be a cv-unqualified object type}}
  (void)sizeof(std::complex<NotDefaultConstructible>);

  // expected-error-re@complex:* {{static assertion failed{{.*}}std::complex<T> requires T to be a cv-unqualified object type}}
  (void)sizeof(std::complex<NotCopyConstructible>);

  // expected-error-re@complex:* {{static assertion failed{{.*}}std::complex<T> requires T to be a cv-unqualified object type}}
  (void)sizeof(std::complex<NotCopyAssignable>);

  // expected-error-re@complex:* {{static assertion failed{{.*}}std::complex<T> requires T to be a cv-unqualified object type}}
  (void)sizeof(std::complex<NotDestructible>);
}
