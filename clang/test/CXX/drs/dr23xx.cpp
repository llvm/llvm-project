// RUN: %clang_cc1 -std=c++98 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors 2>&1 | FileCheck %s
// RUN: %clang_cc1 -std=c++11 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors 2>&1 | FileCheck %s
// RUN: %clang_cc1 -std=c++14 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors 2>&1 | FileCheck %s
// RUN: %clang_cc1 -std=c++17 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors 2>&1 | FileCheck %s
// RUN: %clang_cc1 -std=c++2a %s -verify -fexceptions -fcxx-exceptions -pedantic-errors 2>&1 | FileCheck %s

#if __cplusplus <= 201103L
// expected-no-diagnostics
#endif

namespace dr2353 { // dr2353: 9
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
}

namespace dr2387 { // dr2387: 9
#if __cplusplus >= 201402L
  template<int> int a = 0;
  extern template int a<0>; // ok

  template<int> static int b = 0;
  extern template int b<0>; // expected-error {{internal linkage}}

  template<int> const int c = 0;
  extern template const int c<0>; // ok, has external linkage despite 'const'

  template<typename T> T d = 0;
  extern template int d<int>;
  extern template const int d<const int>;
#endif
}
