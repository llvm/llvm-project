// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-mlir=core %s -o %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

// XFAIL: *

// Note: This is here to track the failed lowering of global pointers. When
//       this test starts passing, the checks should be updated with more
//       specifics.

void *vp;
// CHECK: memref.global "public" @vp

int *ip = 0;
// CHECK: memref.global "public" @ip

double *dp;
// CHECK: memref.global "public" @dp

char **cpp;
// CHECK: memref.global "public" @cpp

void (*fp)();
// CHECK: memref.global "public" @fp

int (*fpii)(int) = 0;
// CHECK: memref.global "public" @fpii

void (*fpvar)(int, ...);
// CHECK: memref.global "public" @fpvar
