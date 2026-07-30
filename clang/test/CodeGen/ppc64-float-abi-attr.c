// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu %s -emit-llvm -mabi=ieeelongdouble -o - | FileCheck %s --check-prefix=IEEE
// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu %s -emit-llvm -mlong-double-64 -o - | FileCheck %s --check-prefix=LDBL64
// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu %s -emit-llvm -DNOLDBL -o - | FileCheck %s --check-prefix=NOLDBL

#ifndef NOLDBL
long double foo(long double a, long double b) {
  return a + b;
}
#endif

int bar() { return 1; }

// CHECK: ![[#]] = !{i32 1, !"float-abi", !"doubledouble"}
// IEEE: ![[#]] = !{i32 1, !"float-abi", !"ieeequad"}
// LDBL64: ![[#]] = !{i32 1, !"float-abi", !"ieeedouble"}
// NOLDBL-NOT: ![[#]] = !{i32 1, !"float-abi"
