//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: FROZEN-CXX03-HEADERS-FIXME

// <valarray>

// template<class T>
// class valarray
//
// T shall be a cv-unqualified object type that satisfies the Cpp17DefaultConstructible,
// Cpp17CopyConstructible, Cpp17CopyAssignable, and Cpp17Destructible requirements (LWG3133)

#include <valarray>

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
  // expected-error-re@valarray:* {{static assertion failed{{.*}}std::valarray<T> requires T to be a cv-unqualified object type}}
  (void)sizeof(std::valarray<const int>);
  // expected-error-re@valarray:* {{static assertion failed{{.*}}std::valarray<T> requires T to be a cv-unqualified object type}}
  (void)sizeof(std::valarray<volatile int>);

  // expected-error-re@valarray:* {{static assertion failed{{.*}}std::valarray<T> requires T to be a cv-unqualified object type}}
  (void)sizeof(std::valarray<NotDefaultConstructible>);

  // expected-error-re@valarray:* {{static assertion failed{{.*}}std::valarray<T> requires T to be a cv-unqualified object type}}
  (void)sizeof(std::valarray<NotCopyConstructible>);

  // expected-error-re@valarray:* {{static assertion failed{{.*}}std::valarray<T> requires T to be a cv-unqualified object type}}
  (void)sizeof(std::valarray<NotCopyAssignable>);

  // expected-error-re@valarray:* {{static assertion failed{{.*}}std::valarray<T> requires T to be a cv-unqualified object type}}
  (void)sizeof(std::valarray<NotDestructible>);
}
