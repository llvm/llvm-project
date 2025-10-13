// RUN: %clang_cc1 -std=c++98 -triple x86_64-unknown-unknown %s -verify=expected -fexceptions -fcxx-exceptions -pedantic-errors -ast-dump | FileCheck %s
// RUN: %clang_cc1 -std=c++11 -triple x86_64-unknown-unknown %s -verify=expected -fexceptions -fcxx-exceptions -pedantic-errors -ast-dump | FileCheck %s
// RUN: %clang_cc1 -std=c++14 -triple x86_64-unknown-unknown %s -verify=expected -fexceptions -fcxx-exceptions -pedantic-errors -ast-dump | FileCheck %s
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-unknown %s -verify=expected -fexceptions -fcxx-exceptions -pedantic-errors -ast-dump | FileCheck %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-unknown %s -verify=expected -fexceptions -fcxx-exceptions -pedantic-errors -ast-dump | FileCheck %s
// RUN: %clang_cc1 -std=c++23 -triple x86_64-unknown-unknown %s -verify=expected -fexceptions -fcxx-exceptions -pedantic-errors -ast-dump | FileCheck %s
// RUN: %clang_cc1 -std=c++2c -triple x86_64-unknown-unknown %s -verify=expected -fexceptions -fcxx-exceptions -pedantic-errors -ast-dump | FileCheck %s

// expected-no-diagnostics

namespace cwg2353 { // cwg2353: 9
  struct X {
    static const int n = 0;
  };

  // CHECK: FunctionDecl {{.*}} use
  int use(X x) {
    // CHECK: MemberExpr {{.*}} .n
    // CHECK-NOT: non_odr_use
    // CHECK: DeclRefExpr {{.*}} 'x'
    // CHECK-NOT: non_odr_use
    return *&x.n;
  }
#pragma clang __debug dump use

  // CHECK: FunctionDecl {{.*}} not_use
  int not_use(X x) {
    // CHECK: MemberExpr {{.*}} .n {{.*}} non_odr_use_constant
    // CHECK: DeclRefExpr {{.*}} 'x'
    return x.n;
  }
#pragma clang __debug dump not_use

  // CHECK: FunctionDecl {{.*}} not_use_2
  int not_use_2(X *x) {
    // CHECK: MemberExpr {{.*}} ->n {{.*}} non_odr_use_constant
    // CHECK: DeclRefExpr {{.*}} 'x'
    return x->n;
  }
#pragma clang __debug dump not_use_2
} // namespace cwg2353
