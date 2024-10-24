// RUN: %clang_cc1 -std=c++2a -verify %s
// RUN: %clang_cc1 -std=c++2a -verify %s -DIMPORT_ERROR=1
// RUN: %clang_cc1 -std=c++2a -verify %s -DIMPORT_ERROR=2

module;

#if IMPORT_ERROR != 2
struct import { struct inner {}; };
#else
// expected-no-diagnostics
#endif
struct module { struct inner {}; };

constexpr int n = 123;

export module m; // #1

// Import errors are fatal, so we test them in isolation.
#if IMPORT_ERROR == 1
import x = {}; // expected-error {{module 'x' not found}}

#elif IMPORT_ERROR == 2
struct X;
template<int> struct import;
template<> struct import<n> {
  static X y;
};

// Well-formed since P1857R3: Modules Dependency Discovery (https://wg21.link/p1857r3),
// it grammatically can't possibly be an import declaration.
struct X {} import<n>::y;

#else
module y = {}; // expected-error {{multiple module declarations}} expected-error 2{{}}
// expected-note@#1 {{previous module declaration}}

::import x = {};
::module y = {};

import::inner xi = {};
module::inner yi = {};

// Ill-formed since P1857R3: Modules Dependency Discovery (https://wg21.link/p1857r3).
namespace N {
  module a; // expected-error {{module declaration can only appear at the top level}}
  import b; // expected-error {{module 'b' not found}}
}

extern "C++" module cxxm;
extern "C++" import cxxi;

template<typename T> module module_var_template;

// This is a variable named 'import' that shadows the type 'import' above.
struct X {} import;
#endif
