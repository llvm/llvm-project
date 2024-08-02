// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fno-clangir-direct-lowering -emit-mlir %s -o %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

float f[32000];
// CHECK: memref.global "public" @f : memref<32000xf32> = dense<0.000000e+00>
double d;
// CHECK: memref.global "public" @d : memref<f64> = dense<0.000000e+00>
float f_init[] = {1.0, 2.0};
// CHECK: memref.global "public" @f_init : memref<2xf32> = dense<[1.000000e+00, 2.000000e+00]>
int i_init[2] = {0, 1};
// CHECK: memref.global "public" @i_init : memref<2xi32> = dense<[0, 1]>
char string[] = "whatnow";
// CHECK: memref.global "public" @string : memref<8xi8> = dense<[119, 104, 97, 116, 110, 111, 119, 0]>
int excess_sint[4] = {1, 2};
// CHECK: memref.global "public" @excess_sint : memref<4xi32> = dense<[1, 2, 0, 0]>
int sint[] = {123, 456, 789};
// CHECK: memref.global "public" @sint : memref<3xi32> = dense<[123, 456, 789]>
