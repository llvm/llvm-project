// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/interface.cppm -o %t/interface.pcm
// RUN: %clang_cc1 -std=c++20 -fmodule-file=Foo=%t/interface.pcm %t/implementation.cppm -verify -DIMPLEMENTATION
// RUN: %clang_cc1 -std=c++20 -fmodule-file=Foo=%t/interface.pcm %t/early_impl.cppm -verify -DEARLY_IMPLEMENTATION
// RUN: %clang_cc1 -std=c++20 -fmodule-file=Foo=%t/interface.pcm %t/user.cppm -verify -DUSER

//--- interface.cppm
// expected-no-diagnostics
module;

template<typename T> struct type_template {
  typedef T type;
  void f(type);
};

template<typename T> void type_template<T>::f(type) {}

template<int = 0, typename = int, template<typename> class = type_template>
struct default_template_args {};

export module Foo;

//--- implementation.cppm
// expected-no-diagnostics
module;

template<typename T> struct type_template {
  typedef T type;
  void f(type);
};

template<typename T> void type_template<T>::f(type) {}

template<int = 0, typename = int, template<typename> class = type_template>
struct default_template_args {};

module Foo;

//--- early_impl.cppm
// expected-no-diagnostics
module;
module Foo;

template<typename T> struct type_template {
  typedef T type;
  void f(type);
};

template<typename T> void type_template<T>::f(type) {}

template<int = 0, typename = int, template<typename> class = type_template>
struct default_template_args {};

//--- user.cppm
// expected-no-diagnostics
import Foo;

template<typename T> struct type_template {
  typedef T type;
  void f(type);
};

template<typename T> void type_template<T>::f(type) {}

template<int = 0, typename = int, template<typename> class = type_template>
struct default_template_args {};
