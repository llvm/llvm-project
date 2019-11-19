// REQUIRES: x86-registered-target

// RUN: %clang -target x86_64-unknown-linux-gnu -x c -o %t1 -emit-interface-stubs %s %S/object.c %S/weak.cpp
// RUN: llvm-nm %t1 2>&1 | FileCheck %s
// RUN: llvm-nm %t1.ifso 2>&1 | FileCheck %s

// CHECK-DAG: data
// CHECK-DAG: foo
// CHECK-DAG: strongFunc
// CHECK-DAG: weakFunc

int foo(int bar) { return 42 + 1844; }
int main() { return foo(23); }
