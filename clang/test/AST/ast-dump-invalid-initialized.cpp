// RUN: not %clang_cc1 -triple x86_64-unknown-unknown -Wno-unused-value -fcxx-exceptions -std=gnu++17 -ast-dump %s | FileCheck -strict-whitespace %s

struct A { A(int, int) {} };
class ForwardDecl;

void test() {
  // CHECK: `-VarDecl {{.*}} a1 'A'
  A a1;
  // CHECK: `-VarDecl {{.*}} a2 'const A'
  const A a2;
  // CHECK: `-VarDecl {{.*}} a3 'A'
  A a3 = garbage();


  // CHECK: `-VarDecl {{.*}} invalid b1 'const A &'
  const A& b1;
  // CHECK: `-VarDecl {{.*}} invalid b2 'ForwardDecl'
  ForwardDecl b2;
}