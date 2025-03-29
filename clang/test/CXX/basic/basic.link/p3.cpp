// RUN: %clang_cc1 -std=c++2a -verify %s
// RUN: %clang_cc1 -std=c++2a -verify %s -DIMPORT_ERROR=1
// RUN: %clang_cc1 -std=c++2a -verify %s -DIMPORT_ERROR=2

module;

#if IMPORT_ERROR != 2
struct import { struct inner {}; };
#endif
struct module { struct inner {}; };

constexpr int n = 123;

export module m; // #1

// Import errors are fatal, so we test them in isolation.
#if IMPORT_ERROR == 1
import x = {}; // expected-error {{expected ';' after module name}}
               // expected-error@-1 {{module 'x' not found}}

#elif IMPORT_ERROR == 2
struct X;
template<int> struct import;
template<> struct import<n> {
  static X y;
};

// This is not valid because the 'import <n>' is a pp-import, even though it
// grammatically can't possibly be an import declaration.
struct X {} import<n>::y; // expected-error {{'n' file not found}}

#else
module y = {}; // expected-error {{multiple module declarations}} expected-error 2{{}}
// expected-note@#1 {{previous module declaration}}

::import x = {};
::module y = {};

import::inner xi = {};
module::inner yi = {};

namespace N {
  module a;
  import b;
}

extern "C++" module cxxm;
extern "C++" import cxxi;

template<typename T> module module_var_template;

// This is a variable named 'import' that shadows the type 'import' above.
struct X {} import;
#endif
