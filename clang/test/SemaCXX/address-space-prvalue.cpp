// RUN: %clang_cc1 %s -ast-dump | FileCheck %s

struct X { int a; };

using GlobalX = X __attribute__((address_space(1)));

GlobalX prvalue();
GlobalX &lvalue();
GlobalX &&xvalue();

void test() {
  // A prvalue should not have an address space even if the function's
  // return type is address-space qualified.
  // CHECK: VarDecl {{.*}} v 'X'
  // CHECK: CallExpr {{.*}} 'X'{{$}}
  auto v = prvalue();

  // CHECK: VarDecl {{.*}} l '__attribute__((address_space(1))) X &'
  // CHECK: CallExpr {{.*}}:'__attribute__((address_space(1))) X' lvalue
  auto &l = lvalue();

  // CHECK: VarDecl {{.*}} r '__attribute__((address_space(1))) X &&'
  // CHECK: CallExpr {{.*}}:'__attribute__((address_space(1))) X' xvalue
  auto &&r = xvalue();
}
