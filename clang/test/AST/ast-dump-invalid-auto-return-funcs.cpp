// RUN: not %clang_cc1 -triple x86_64-unknown-unknown -Wno-unused-value -fcxx-exceptions -frecovery-ast -std=gnu++17 -ast-dump %s | FileCheck -strict-whitespace %s

// CHECK: FunctionDecl {{.*}} s1 'auto ()'
auto s1(); // valid
// FIXME: why we're finding int as the return type. int is used as a fallback type?
// CHECK: FunctionDecl {{.*}} invalid s2 'auto () -> int'
auto s2() -> undef();
// CHECK: FunctionDecl {{.*}} invalid s3 'auto () -> int'
auto s3() -> decltype(undef());
// CHECK: FunctionDecl {{.*}} invalid s4 'auto ()'
auto s4() {
  return undef();
}
// CHECK: FunctionDecl {{.*}} s5 'void ()'
auto s5() {} // valid, no return stmt, fallback to void

class Foo {
  // CHECK: CXXMethodDecl {{.*}} foo1 'auto ()'
  auto foo1(); // valid
  // CHECK: CXXMethodDecl {{.*}} invalid foo2 'auto () -> int'
  auto foo2() -> undef();
  // CHECK: CXXMethodDecl {{.*}} invalid foo3 'auto () -> int'
  auto foo3() -> decltype(undef());
  // CHECK: CXXMethodDecl {{.*}} invalid foo4 'auto ()'
  auto foo4() { return undef(); }
  // CHECK: CXXMethodDecl {{.*}} foo5 'void ()'
  auto foo5() {} // valid, no return stmt, fallback to void.
};
