// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -x c++ -std=c++20 -fmodules -fmodule-name=foo %t/foo.map -emit-module -o %t/foo.pcm
// RUN: %clang_cc1 -x c++ -std=c++20 -fmodules -fmodules-cache-path=%t \
// RUN:     -fmodule-file=%t/foo.pcm %t/use.cpp -verify -fsyntax-only

//--- foo.map
module "foo" {
  export * 
  header "foo.h"
}

//--- foo.h
template<class T1, int V = 0>
class A;

template <typename T>
class templ_params {};

template<class T1, template <typename> typename T2 = templ_params>
class B;

template<class T1, class T2 = int>
class C;

//--- use.cpp
#include "foo.h"
template<class T1, int V = 1> // expected-error {{template parameter default argument is inconsistent with previous definition}}
class A;   // expected-note@foo.h:1 {{previous default template argument defined in module foo}}

template <typename T>
class templ_params2 {};

template<class T1, template <typename> typename T2 = templ_params2> // expected-error {{template parameter default argument is inconsistent with previous definition}}
class B; // expected-note@foo.h:7 {{previous default template argument defined in module foo}}

template<class T1, class T2 = double> // expected-error {{template parameter default argument is inconsistent with previous definition}}
class C; // expected-note@foo.h:10 {{previous default template argument defined in module foo}}
