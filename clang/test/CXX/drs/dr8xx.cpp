// RUN: %clang_cc1 -std=c++98 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors 2>&1 | FileCheck %s
// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors 2>&1 | FileCheck %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors 2>&1 | FileCheck %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors 2>&1 | FileCheck %s
// RUN: %clang_cc1 -std=c++23 -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors 2>&1 | FileCheck %s
// RUN: %clang_cc1 -std=c++2c -triple x86_64-unknown-unknown %s -verify -fexceptions -fcxx-exceptions -pedantic-errors 2>&1 | FileCheck %s

// expected-no-diagnostics

namespace dr873 { // dr873: yes
#if __cplusplus >= 201103L
template <typename T> void f(T &&);
template <> void f(int &) {}  // #1
template <> void f(int &&) {} // #2
void g(int i) {
  f(i); // calls f<int&>(int&), i.e., #1
#pragma clang __debug dump f(i)
  //      CHECK: CallExpr {{.*}}
  // CHECK-NEXT: |-ImplicitCastExpr {{.*}}
  // CHECK-NEXT: | `-DeclRefExpr {{.*}} 'f' 'void (int &)' {{.*}}

  f(0); // calls f<int>(int&&), i.e., #2
#pragma clang __debug dump f(0)
  //      CHECK: CallExpr {{.*}}
  // CHECK-NEXT: |-ImplicitCastExpr {{.*}}
  // CHECK-NEXT: | `-DeclRefExpr {{.*}} 'f' 'void (int &&)' {{.*}}
}
#endif
} // namespace dr873
