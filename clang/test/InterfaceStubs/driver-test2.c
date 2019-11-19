// REQUIRES: x86-registered-target

// RUN: %clang -target x86_64-unknown-linux-gnu -c -emit-interface-stubs \
// RUN:   %s %S/object.c %S/weak.cpp
// RUN: %clang -emit-interface-stubs -emit-merged-ifs \
// RUN:   driver-test2.o object.o weak.o -S -o - | FileCheck %s

// CHECK-DAG: data
// CHECK-DAG: bar
// CHECK-DAG: strongFunc
// CHECK-DAG: weakFunc

int bar(int a) { return a; }
int main() { return 0; }
