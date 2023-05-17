// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -x c++ -std=c++20 -fmodules -fmodules-cache-path=%t -fmodule-map-file=%t/module.map %t/foo.cpp -verify

//--- module.map
module "foo" {
  export * 
  header "foo.h"
}
module "bar" {
  export * 
  header "bar.h"
}

//--- foo.h
template <class T>
concept A = true;

//--- bar.h
template <class T>
concept A = false;

//--- foo.cpp
#include "bar.h"
#include "foo.h"

template <class T> void foo() requires A<T> {}  // expected-error 1+{{reference to 'A' is ambiguous}}
                                                // expected-note@* 1+{{candidate found by name lookup}}

int main() {
  foo<int>();
  return 0;
}
