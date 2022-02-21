// This tests for [basic.lookup.argdep]/p4.2:
//   Argument-dependent lookup finds all declarations of functions and function templates that
// - ...
// - are declared as a friend ([class.friend]) of any class with a reachable definition in the set of associated entities,
//
// RUN: rm -fr %t
// RUN: mkdir %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/Friend-in-reachable-class.cppm -o %t/X.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/Use.cpp -verify -fsyntax-only
//
//--- Friend-in-reachable-class.cppm
module;
# 3 __FILE__ 1
struct A {
  friend int operator+(const A &lhs, const A &rhs) {
    return 0;
  }
};
# 6 "" 2
export module X;
export using ::A;

//--- Use.cpp
// expected-no-diagnostics
import X;
int use() {
  A a, b;
  return a + b;
}
