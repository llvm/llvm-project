// Test this without pch.
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -DBODY %s -o - | FileCheck %s

// Test with pch.
// RUN: %clang_cc1 -emit-pch -o %t %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -include-pch %t -DBODY %s -o - | FileCheck %s

// RUN: %clang_cc1 -emit-pch -fpch-instantiate-templates -o %t %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -include-pch %t -DBODY %s -o - | FileCheck %s

// expected-no-diagnostics

#ifndef HEADER_H
#define HEADER_H
struct A {
  void foo() { bar<0>(); } // This will trigger implicit instantiation of bar<0>() in the PCH.
  template <int N>
  void bar();
};
#endif

#ifdef BODY
// But the definition is only in the source, so the instantiation must be delayed until the TU.
template <int N>
void A::bar() {}

void test(A *a) { a->foo(); }
#endif

// CHECK: define linkonce_odr void @_ZN1A3barILi0EEEvv
