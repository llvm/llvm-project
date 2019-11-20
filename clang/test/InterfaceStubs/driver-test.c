// REQUIRES: x86-registered-target
// REQUIRES: shell

// RUN: mkdir -p %t; cd %t
// RUN: %clang -target x86_64-unknown-linux-gnu -x c -S -emit-interface-stubs %s %S/object.c %S/weak.cpp && \
// RUN: llvm-nm %t/a.out.ifso 2>&1 | FileCheck --check-prefix=CHECK-IFS %s

// CHECK-IFS-DAG: data
// CHECK-IFS-DAG: foo
// CHECK-IFS-DAG: strongFunc
// CHECK-IFS-DAG: weakFunc

int foo(int bar) { return 42 + 1844; }
