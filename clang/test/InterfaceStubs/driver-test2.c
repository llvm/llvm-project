// REQUIRES: x86-registered-target
// REQUIRES: shell

// RUN: mkdir -p %t; cd %t
// RUN: %clang -target x86_64-unknown-linux-gnu -c -emit-interface-stubs \
// RUN:   %s %S/object.c %S/weak.cpp
// RUN: %clang -emit-interface-stubs -emit-merged-ifs \
// RUN:   %t/driver-test2.o %t/object.o %t/weak.o -S -o - 2>&1 | FileCheck %s

// CHECK-DAG: data
// CHECK-DAG: bar
// CHECK-DAG: strongFunc
// CHECK-DAG: weakFunc

int bar(int a) { return a; }
int main() { return 0; }
