// RUN: %clang_cc1 -w -triple x86_64-apple-darwin10 -emit-llvm -o - %s | FileCheck %s
// rdar://11759609

// At or below target max alignment with no aligned attribute should align based
// on the size of vector.
double __attribute__((vector_size(16))) v1;
// CHECK: @v1 {{.*}}, align 16
double __attribute__((vector_size(32))) v2;
// CHECK: @v2 {{.*}}, align 32

// Alignment above target max alignment with no aligned attribute should align
// based on the target max.
double __attribute__((vector_size(64))) v3;
// CHECK: @v3 {{.*}}, align 32
double __attribute__((vector_size(1024))) v4;
// CHECK: @v4 {{.*}}, align 32

// Aliged attribute should always override.
double __attribute__((vector_size(16), aligned(16))) v5;
// CHECK: @v5 {{.*}}, align 16
double __attribute__((vector_size(16), aligned(64))) v6;
// CHECK: @v6 {{.*}}, align 64
double __attribute__((vector_size(32), aligned(16))) v7;
// CHECK: @v7 {{.*}}, align 16
double __attribute__((vector_size(32), aligned(64))) v8;
// CHECK: @v8 {{.*}}, align 64

// Check non-power of 2 widths.
double __attribute__((vector_size(24))) v9;
// CHECK: @v9 {{.*}}, align 32
double __attribute__((vector_size(40))) v10;
// CHECK: @v10 {{.*}}, align 32

// Check non-power of 2 widths with aligned attribute.
double __attribute__((vector_size(24), aligned(64))) v11;
// CHECK: @v11 {{.*}}, align 64
double __attribute__((vector_size(80), aligned(16))) v12;
// CHECK: @v12 {{.*}}, align 16
