// RUN: clang %s -g -c -o %t --target=x86_64-apple-macosx
// RUN: lldb-test symbols --name=A::foo --find=variable %t | FileCheck %s

// CHECK: Found 1 variables:

struct A {
  static int foo;
};
int A::foo;
// NAME-DAG: name = "foo", {{.*}} decl = find-qualified-variable.cpp:[[@LINE-1]]

struct B {
  static int foo;
};
int B::foo;
