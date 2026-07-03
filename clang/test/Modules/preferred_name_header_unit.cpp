// RUN: rm -fR %t
// RUN: split-file %s %t
// RUN: cd %t
// RUN: %clang_cc1 -verify -w -std=c++20 -fmodule-name=h1.h -emit-header-unit -xc++-user-header h1.h -o h1.pcm
// RUN: %clang_cc1 -verify -w -std=c++20 -fmodule-map-file=module.modulemap -fmodule-file=h1.h=h1.pcm main.cpp -o main.o

//--- module.modulemap
module "h1.h" {
  header "h1.h"
  export *
}

//--- h0.h
// expected-no-diagnostics
#pragma once
namespace std {

template <class _CharT, class = _CharT, class = _CharT> class basic_string;

namespace pmr {
using string = basic_string<char>;
}

template <class, class, class>
class __attribute__((__preferred_name__(pmr::string))) basic_string;

template <class> class basic_string_view {};

template <class _CharT, class _Traits, class _Allocator> class basic_string {
  typedef _CharT value_type;
  typedef _Allocator allocator_type;
  struct __rep;
public:
  template <class _Tp>
  basic_string(_Tp) {}
  basic_string operator+=(value_type);
};

namespace filesystem {
class path {
  typedef char value_type;
  value_type preferred_separator;
  typedef basic_string<value_type> string_type;
  typedef basic_string_view<value_type> __string_view;
  template <class _Source> void append(_Source) {
    __pn_ += preferred_separator;
  }
  void __root_directory() { append(string_type(__string_view{})); }
  string_type __pn_;
};
} // namespace filesystem
} // namespace std

//--- h1.h
// expected-no-diagnostics
#pragma once

#include "h0.h"

//--- main.cpp
// expected-no-diagnostics
#include "h0.h"

import "h1.h";
