// RUN: rm -f %t
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -emit-module-interface %s -o %t -DINTERFACE
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -fmodule-file=%t %s -verify -DIMPLEMENTATION
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -fmodule-file=%t %s -verify -DEARLY_IMPLEMENTATION
// RUN: %clang_cc1 -std=c++1z -fmodules-ts -fmodule-file=%t %s -verify -DUSER

// expected-no-diagnostics

#ifdef USER
import Foo;
#endif

#ifdef EARLY_IMPLEMENTATION
module Foo;
#endif

template<typename T> struct type_template {
  typedef T type;
  void f(type);
};

template<typename T> void type_template<T>::f(type) {}

template<int = 0, typename = int, template<typename> class = type_template>
struct default_template_args {};

#ifdef INTERFACE
export module Foo;
#endif

#ifdef IMPLEMENTATION
module Foo;
#endif
