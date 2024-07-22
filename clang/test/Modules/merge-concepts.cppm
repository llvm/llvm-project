// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -emit-module-interface -I%t %t/A.cppm -o %t/A.pcm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface -I%t %t/B.cppm -o %t/B.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/Use.cpp -verify -fsyntax-only
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/Use2.cpp -verify -fsyntax-only
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/Use3.cpp -verify -fsyntax-only
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/Use4.cpp -verify -fsyntax-only
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/C.cppm -verify -fsyntax-only
// RUN: %clang_cc1 -std=c++20 -I%t %t/D.cppm -verify -fsyntax-only
// RUN: %clang_cc1 -std=c++20 -I%t %t/E.cppm -verify -fsyntax-only
// RUN: %clang_cc1 -std=c++20 -I%t %t/F.cppm -verify -fsyntax-only
//
// Testing header units for coverity.
// RUN: %clang_cc1 -std=c++20 -emit-header-unit -xc++-user-header %t/foo.h -o %t/foo.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t -fmodule-file=%t/foo.pcm %t/Use5.cpp -verify -fsyntax-only
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t -fmodule-file=%t/foo.pcm %t/Use6.cpp -verify -fsyntax-only
//
// Testing with module map modules. It is unclear about the relation ship between Clang modules and
// C++20 Named Modules. Will they coexist? Or will they be mutually exclusive?
// The test here is for primarily coverity.
//
// RUN: rm -f %t/foo.pcm
// RUN: %clang_cc1 -std=c++20 -fmodules -fmodules-cache-path=%t -fprebuilt-module-path=%t \
// RUN:   -fmodule-map-file=%t/module.modulemap %t/Use7.cpp -verify -fsyntax-only
// RUN: %clang_cc1 -std=c++20 -fmodules -fmodules-cache-path=%t -fprebuilt-module-path=%t \
// RUN:   -fmodule-map-file=%t/module.modulemap %t/Use7.cpp -verify -fsyntax-only
// Testing module map modules with named modules.
// RUN: %clang_cc1 -std=c++20 -fmodules -fmodules-cache-path=%t -fmodule-map-file=%t/module.modulemap \
// RUN:   %t/A.cppm -o %t/A.pcm
// RUN: %clang_cc1 -std=c++20 -fmodules -fmodules-cache-path=%t -fprebuilt-module-path=%t \
// RUN:   -fmodule-map-file=%t/module.modulemap %t/Use7.cpp -verify -fsyntax-only
// RUN: %clang_cc1 -std=c++20 -fmodules -fmodules-cache-path=%t -fprebuilt-module-path=%t \
// RUN:   -fmodule-map-file=%t/module.modulemap %t/Use7.cpp -verify -fsyntax-only

//
//--- foo.h
#ifndef FOO_H
#define FOO_H
template <class T, class U>
concept same_as = __is_same(T, U);
#endif

// The compiler would warn if we include foo_h twice without guard.
//--- redecl.h
#ifndef REDECL_H
#define REDECL_H
template <class T, class U>
concept same_as = __is_same(T, U);
#endif

//--- A.cppm
module;
#include "foo.h"
export module A;
export using ::same_as;

//--- B.cppm
module;
#include "foo.h"
export module B;
export using ::same_as;

//--- Use.cpp
// expected-no-diagnostics
import A;
import B;

template <class T> void foo()
  requires same_as<T, int>
{}

//--- Use2.cpp
// expected-no-diagnostics
#include "foo.h"
import A;

template <class T> void foo()
  requires same_as<T, int>
{}

//--- Use3.cpp
// expected-no-diagnostics
import A;
#include "foo.h"

template <class T> void foo()
  requires same_as<T, int>
{}

//--- Use4.cpp
// expected-no-diagnostics
import A;
import B;
#include "foo.h"

template <class T> void foo()
  requires same_as<T, int>
{}

//--- C.cppm
// expected-no-diagnostics
module;
#include "foo.h"
export module C;
import A;
import B;

template <class T> void foo()
  requires same_as<T, int>
{}

//--- D.cppm
module;
#include "foo.h"
#include "redecl.h"
export module D;
export using ::same_as;

// expected-error@* {{redefinition of 'same_as'}}
// expected-note@* 1+{{previous definition is here}}

//--- E.cppm
module;
#include "foo.h"
export module E;
export template <class T, class U>
concept same_as = __is_same(T, U);

// expected-error@* {{redefinition of 'same_as'}}
// expected-note@* 1+{{previous definition is here}}

//--- F.cppm
export module F;
template <class T, class U>
concept same_as = __is_same(T, U);
template <class T, class U>
concept same_as = __is_same(T, U);

// expected-error@* {{redefinition of 'same_as'}}
// expected-note@* 1+{{previous definition is here}}

//--- Use5.cpp
import "foo.h";  // expected-warning {{the implementation of header units is in an experimental phase}}
import A;

template <class T> void foo()
  requires same_as<T, int>
{}

//--- Use6.cpp
import A;
import "foo.h"; // expected-warning {{the implementation of header units is in an experimental phase}}

template <class T> void foo()
  requires same_as<T, int>
{}

//--- module.modulemap
module "foo" {
  export * 
  header "foo.h"
}

//--- Use7.cpp
// expected-no-diagnostics
#include "foo.h"
import A;

template <class T> void foo()
  requires same_as<T, int>
{}

//--- Use8.cpp
// expected-no-diagnostics
import A;
#include "foo.h"

template <class T> void foo()
  requires same_as<T, int>
{}
