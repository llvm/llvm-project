// RUN: %clang_cc1 -ast-list %s | FileCheck -strict-whitespace %s

template <typename>
struct Tpl {
  template <typename>
  static int var;
};
// CHECK: Tpl
// CHECK-NEXT: Tpl::(anonymous)
// CHECK-NEXT: Tpl
// CHECK-NEXT: Tpl::var
// CHECK-NEXT: Tpl::(anonymous)
// CHECK-NEXT: Tpl::var

template <typename T>
template <typename>
int Tpl<T>::var;
// CHECK-NEXT: Tpl::var
// CHECK-NEXT: Tpl::(anonymous)
// CHECK-NEXT: Tpl::var
// CHECK-NEXT: T

int i = Tpl<int>::var<int>;
// CHECK-NEXT: i
// CHECK-NEXT: Tpl<int>::var
