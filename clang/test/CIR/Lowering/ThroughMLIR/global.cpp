// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fno-clangir-direct-lowering -emit-mlir %s -o %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

float f[32000];
double d;

// CHECK: memref.global "public" @f : memref<32000xf32> = dense<0.000000e+00>
// CHECK: memref.global "public" @d : memref<f64> = dense<0.000000e+00>
