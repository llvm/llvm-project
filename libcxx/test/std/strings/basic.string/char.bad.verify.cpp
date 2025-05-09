//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: FROZEN-CXX03-HEADERS-FIXME

// <string>
//   ... manipulating sequences of any non-array trivial standard-layout types.

#include <string>
#include <type_traits>
#include "test_traits.h"

struct NotTriviallyCopyable {
  int value;
  NotTriviallyCopyable& operator=(const NotTriviallyCopyable& other) {
    value = other.value;
    return *this;
  }
};

struct NotTriviallyDefaultConstructible {
  NotTriviallyDefaultConstructible() : value(3) {}
  int value;
};

struct NotStandardLayout {
public:
  NotStandardLayout() : one(1), two(2) {}
  int sum() const { return one + two; } // silences "unused field 'two' warning"
  int one;

private:
  int two;
};

void f() {
  {
    // array
    typedef char C[3];
    static_assert(std::is_array<C>::value, "");
    std::basic_string<C, test_traits<C> > s;
    // expected-error-re@*:* {{static assertion failed{{.*}}Character type of basic_string must not be an array}}
  }

  {
    // not trivially copyable
    static_assert(!std::is_trivially_copyable<NotTriviallyCopyable>::value, "");
    std::basic_string<NotTriviallyCopyable, test_traits<NotTriviallyCopyable> > s;
    // expected-error-re@*:* {{static assertion failed{{.*}}Character type of basic_string must be trivially copyable}}
  }

  {
    // not trivially default constructible
    static_assert(!std::is_trivially_default_constructible<NotTriviallyDefaultConstructible>::value, "");
    std::basic_string<NotTriviallyDefaultConstructible, test_traits<NotTriviallyDefaultConstructible> > s;
    // expected-error-re@*:* {{static assertion failed{{.*}}Character type of basic_string must be trivially default constructible}}
  }

  {
    // not standard layout
    static_assert(!std::is_standard_layout<NotStandardLayout>::value, "");
    std::basic_string<NotStandardLayout, test_traits<NotStandardLayout> > s;
    // expected-error-re@*:* {{static assertion failed{{.*}}Character type of basic_string must be standard-layout}}
  }
}
