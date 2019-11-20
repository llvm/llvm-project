// REQUIRES: x86-registered-target
// REQUIRES: !system-darwin && !system-windows

// RUN: %clang -o %t1 -target x86_64-unknown-linux-gnu \
// RUN:   -emit-interface-stubs -emit-merged-ifs %s %S/object.c %S/weak.cpp
// RUN: cat %t1.ifs | FileCheck %s

// CHECK-DAG: data
// CHECK-DAG: foo
// CHECK-DAG: strongFunc
// CHECK-DAG: weakFunc

int foo(int bar) { return 42 + 1844; }
int main() { return foo(23); }
