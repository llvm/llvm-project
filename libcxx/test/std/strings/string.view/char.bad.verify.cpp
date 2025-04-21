//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: !stdlib=libc++ && (c++03 || c++11 || c++14)

// XFAIL: FROZEN-CXX03-HEADERS-FIXME

// <string_view>
//   ... manipulating sequences of any non-array trivial standard-layout types.

#include <string>
#include <type_traits>
#include "../basic.string/test_traits.h"

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

int main(int, char**) {
  {
    //  array
    typedef char C[3];
    static_assert(std::is_array<C>::value, "");
    std::basic_string_view<C, test_traits<C> > sv;
    //  expected-error-re@string_view:* {{static assertion failed{{.*}}Character type of basic_string_view must not be an array}}
  }

  {
    //  not trivially copyable
    static_assert(!std::is_trivially_copyable<NotTriviallyCopyable>::value, "");
    std::basic_string_view<NotTriviallyCopyable, test_traits<NotTriviallyCopyable> > s;
    // expected-error-re@*:* {{static assertion failed{{.*}}Character type of basic_string_view must be trivially copyable}}
  }

  {
    //  not trivially default constructible
    static_assert(!std::is_trivially_default_constructible<NotTriviallyDefaultConstructible>::value, "");
    std::basic_string_view<NotTriviallyDefaultConstructible, test_traits<NotTriviallyDefaultConstructible> > sv;
    //  expected-error-re@string_view:* {{static assertion failed{{.*}}Character type of basic_string_view must be trivially default constructible}}
  }

  {
    //  not standard layout
    static_assert(!std::is_standard_layout<NotStandardLayout>::value, "");
    std::basic_string_view<NotStandardLayout, test_traits<NotStandardLayout> > sv;
    //  expected-error-re@string_view:* {{static assertion failed{{.*}}Character type of basic_string_view must be standard-layout}}
  }

  return 0;
}
