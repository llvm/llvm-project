// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 -verify %t/M.cppm
// RUN: %clang_cc1 -std=c++20 -verify %t/ImportError1.cppm
// RUN: %clang_cc1 -std=c++20 -verify %t/ImportError2.cppm
// RUN: %clang_cc1 -std=c++20 -Wno-reserved-module-identifier -emit-module-interface %t/std.cppm -o %t/std.pcm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/A.cppm -o %t/A.pcm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/A-B.cppm -o %t/A-B.pcm
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify -fmodule-file=std=%t/std.pcm -fmodule-file=A=%t/A.pcm -fmodule-file=A:B=%t/A-B.pcm %t/A_impl.cppm
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify -fmodule-file=A=%t/A.pcm %t/User.cppm 

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
  module a;
  import b;
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
module;

struct module { struct inner {}; };

constexpr int n = 123;

export module m; // #1

struct X;
template<int> struct import;
template<> struct import<n> {
  static X y;
};

// This is not valid because the 'import <n>' is a pp-import, even though it
// grammatically can't possibly be an import declaration.
struct X {} import<n>::y; // expected-error {{'n' file not found}}

//--- A.cppm
export module A;
const double delta=0.01;
export {
  template<class F>
  double derivative(F &&f,double x) {
    return (f(x+delta)-f(x))/delta;
  }
}

//--- std.cppm
export module std;
export using size_t = decltype(sizeof(void *));

export namespace std {
  template <typename T, size_t N>
  struct array {};
}

//--- A-B.cppm
module A:B;
const int dimensions=3;

//--- A_impl.cppm
// expected-no-diagnostics
module A;
import std;
import :B;

using vector = std::array<double, dimensions>;  // error: lookup failed until P2788R0(Linkage for modular constants).

//--- User.cppm
// expected-no-diagnostics
import A;
double d=derivative([](double x) {return x*x;},2);  // error: names delta until P2788R0(Linkage for modular constants).
