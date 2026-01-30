// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 -verify %t/M.cpp
// RUN: %clang_cc1 -std=c++20 -verify %t/ImportError1.cpp
// RUN: %clang_cc1 -std=c++20 -verify %t/ImportError2.cpp

//--- M.cpp
module;

struct import { struct inner {}; };
struct module { struct inner {}; };
constexpr int n = 123;

export module m; // #1
module y = {}; // expected-error {{multiple module declarations}}
// expected-error@-1 {{unexpected preprocessing token '=' after module name, only ';' and '[' (start of attribute specifier sequence) are allowed}}
// expected-note@#1 {{previous module declaration}}

::import x = {};
::module y = {};

import::inner xi = {};
module::inner yi = {};

namespace N {
  module a; // expected-error {{module declaration can only appear at the top level}}
  import b; // expected-error {{import declaration can only appear at the top level}}
}

extern "C++" module cxxm;
extern "C++" import cxxi;

template<typename T> module module_var_template;

// This is a variable named 'import' that shadows the type 'import' above.
struct X {} import;

//--- ImportError1.cpp
module;

struct import { struct inner {}; };
struct module { struct inner {}; };

constexpr int n = 123;

export module m; // #1

import x = {}; // expected-error {{import directive must end with a ';'}}
               // expected-error@-1 {{module 'x' not found}}

//--- ImportError2.cpp
// expected-no-diagnostics
module;

struct module { struct inner {}; };

constexpr int n = 123;

export module m; // #1

struct X;
template<int> struct import;
template<> struct import<n> {
  static X y;
};

struct X {} import<n>::y;
