// REQUIRES: x86-registered-target

// NOTE: -fno-integrated-cc1 has been added to work around an ASAN failure
//       caused by in-process cc1 invocation. Clang InterfaceStubs is not the
//       culprit, but Clang Interface Stubs' Driver pipeline setup uncovers an
//       existing ASAN issue when invoking multiple normal cc1 jobs along with
//       multiple Clang Interface Stubs cc1 jobs together.
//       There is currently a discussion of this going on at:
//         https://reviews.llvm.org/D69825
// RUN: mkdir -p %t; cd %t
// RUN: %clang -target x86_64-unknown-linux-gnu -c -emit-interface-stubs \
// RUN:   -fno-integrated-cc1 \
// RUN:   %s %S/object.c %S/weak.cpp
// RUN: %clang -emit-interface-stubs -emit-merged-ifs \
// RUN:   -fno-integrated-cc1 \
// RUN:   %t/driver-test2.o %t/object.o %t/weak.o -S -o - 2>&1 | FileCheck %s

// CHECK-DAG: data
// CHECK-DAG: bar
// CHECK-DAG: strongFunc
// CHECK-DAG: weakFunc

int bar(int a) { return a; }
int main() { return 0; }
