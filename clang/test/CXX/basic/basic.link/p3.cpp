// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++2a -verify %t/M.cppm
// RUN: %clang_cc1 -std=c++2a -verify %t/ImportError1.cppm
// RUN: %clang_cc1 -std=c++2a -verify %t/ImportError2.cppm

//--- M.cppm
module;

struct import { struct inner {}; };
struct module { struct inner {}; };
constexpr int n = 123;

export module m; // #1
module y = {}; // expected-error {{multiple module declarations}} expected-error 2{{}}
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

//--- ImportError1.cppm
module;

struct import { struct inner {}; };
struct module { struct inner {}; };

constexpr int n = 123;

export module m; // #1

import x = {}; // expected-error {{expected ';' after module name}}
               // expected-error@-1 {{module 'x' not found}}

//--- ImportError2.cppm
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
