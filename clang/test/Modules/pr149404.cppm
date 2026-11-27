// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 -emit-module-interface -o %t/format.pcm %t/format.cppm
// RUN: %clang_cc1 -std=c++20  -emit-module-interface -o %t/includes_in_gmf.pcm %t/includes_in_gmf.cppm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/test.cpp -fsyntax-only -verify

//--- format.h
#pragma once

namespace test {

template <class _Tp>
struct type_identity {
    typedef _Tp type;
};

template <class> struct formatter;
template <> struct formatter<char> {};

template <class T, class>
struct basic_format_string {
  static inline const int __handles_ = [] {
    static_assert(__is_same(T, char));
    return 0;
  }();

  basic_format_string(const T*) {
    static_assert(__is_same(T, char));
    (void)__handles_;
  }
};

template <class T>
void format(basic_format_string<char, typename type_identity<T>::type>, T) {}

template <class T>
void format(basic_format_string<wchar_t, typename type_identity<T>::type>, T) = delete;
}

//--- format.cppm
module;
#include "format.h"
export module format;

export namespace test {
  using test::format;
}

void something() {
  test::format("", 0);
}

//--- includes_in_gmf.cppm
module;
#include "format.h"
export module includes_in_gmf;

//--- test.cpp
// expected-no-diagnostics

#include "format.h" // Uncomment this to get the good version.
import format;
import includes_in_gmf;

void f() {
  test::format("", 0);
  test::format("", 'r');
}
