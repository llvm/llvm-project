// REQUIRES: x86-registered-target

// NOTE: -fno-integrated-cc1 has been added to work around an ASAN failure
//       caused by in-process cc1 invocation. Clang InterfaceStubs is not the
//       culprit, but Clang Interface Stubs' Driver pipeline setup uncovers an
//       existing ASAN issue when invoking multiple normal cc1 jobs along with
//       multiple Clang Interface Stubs cc1 jobs together.
//       There is currently a discussion of this going on at:
//         https://reviews.llvm.org/D69825
// RUN: mkdir -p %t; cd %t
// RUN: %clang -target x86_64-unknown-linux-gnu -x c -S \
// RUN:   -fno-integrated-cc1 \
// RUN: -emit-interface-stubs %s %S/object.c %S/weak.cpp && \
// RUN: llvm-nm -D %t/a.out.ifso 2>&1 | FileCheck --check-prefix=CHECK-IFS %s

// CHECK-IFS-DAG: data
// CHECK-IFS-DAG: foo
// CHECK-IFS-DAG: strongFunc
// CHECK-IFS-DAG: weakFunc

int foo(int bar) { return 42 + 1844; }
