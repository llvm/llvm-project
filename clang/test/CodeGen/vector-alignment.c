// RUN: %clang_cc1 -w -triple x86_64-apple-darwin10 \
// RUN:  -emit-llvm -o - %s | FileCheck %s --check-prefix=ALL --check-prefix=SSE
// RUN: %clang_cc1 -w -triple   i386-apple-darwin10 \
// RUN:  -emit-llvm -o - %s | FileCheck %s --check-prefix=ALL --check-prefix=SSE
// RUN: %clang_cc1 -w -triple x86_64-apple-darwin10 -target-feature +avx \
// RUN:  -emit-llvm -o - %s | FileCheck %s --check-prefix=ALL --check-prefix=AVX
// RUN: %clang_cc1 -w -triple   i386-apple-darwin10 -target-feature +avx \
// RUN:  -emit-llvm -o - %s | FileCheck %s --check-prefix=ALL --check-prefix=AVX
// RUN: %clang_cc1 -w -triple x86_64-apple-darwin10 -target-feature +avx512f \
// RUN:  -emit-llvm -o - %s | FileCheck %s --check-prefix=ALL --check-prefix=AVX512
// RUN: %clang_cc1 -w -triple   i386-apple-darwin10 -target-feature +avx512f \
// RUN:  -emit-llvm -o - %s | FileCheck %s --check-prefix=ALL --check-prefix=AVX512
// rdar://11759609

// At or below target max alignment with no aligned attribute should align based
// on the size of vector.
double __attribute__((vector_size(16))) v1;
// SSE: @v1 {{.*}}, align 16
// AVX: @v1 {{.*}}, align 16
// AVX512: @v1 {{.*}}, align 16
double __attribute__((vector_size(32))) v2;
// SSE: @v2 {{.*}}, align 16
// AVX: @v2 {{.*}}, align 32
// AVX512: @v2 {{.*}}, align 32
typedef __attribute__((__ext_vector_type__(16))) _Bool v2b_type;
v2b_type v2b;
// ALL: @v2b {{.*}}, align 2

// Alignment above target max alignment with no aligned attribute should align
// based on the target max.
double __attribute__((vector_size(64))) v3;
// SSE: @v3 {{.*}}, align 16
// AVX: @v3 {{.*}}, align 32
// AVX512: @v3 {{.*}}, align 64
double __attribute__((vector_size(1024))) v4;
// SSE: @v4 {{.*}}, align 16
// AVX: @v4 {{.*}}, align 32
// AVX512: @v4 {{.*}}, align 64
typedef __attribute__((__ext_vector_type__(8192))) _Bool v4b_type;
v4b_type v4b;
// SSE: @v4b {{.*}}, align 16
// AVX: @v4b {{.*}}, align 32
// AVX512: @v4b {{.*}}, align 64

// Aliged attribute should always override.
double __attribute__((vector_size(16), aligned(16))) v5;
// ALL: @v5 {{.*}}, align 16
double __attribute__((vector_size(16), aligned(64))) v6;
// ALL: @v6 {{.*}}, align 64
double __attribute__((vector_size(32), aligned(16))) v7;
// ALL: @v7 {{.*}}, align 16
double __attribute__((vector_size(32), aligned(64))) v8;
// ALL: @v8 {{.*}}, align 64
typedef __attribute__((ext_vector_type(256), aligned(128))) _Bool v8b_type;
v8b_type v8b;
// ALL: @v8b {{.*}}, align 128

// Check non-power of 2 widths.
double __attribute__((vector_size(24))) v9;
// SSE: @v9 {{.*}}, align 16
// AVX: @v9 {{.*}}, align 32
// AVX512: @v9 {{.*}}, align 32
double __attribute__((vector_size(40))) v10;
// SSE: @v10 {{.*}}, align 16
// AVX: @v10 {{.*}}, align 32
// AVX512: @v10 {{.*}}, align 64
typedef __attribute__((ext_vector_type(248))) _Bool v10b_type;
v10b_type v10b;
// SSE: @v10b {{.*}}, align 16
// AVX: @v10b {{.*}}, align 32
// AVX512: @v10b {{.*}}, align 32

// Check non-power of 2 widths with aligned attribute.
double __attribute__((vector_size(24), aligned(64))) v11;
// ALL: @v11 {{.*}}, align 64
double __attribute__((vector_size(80), aligned(16))) v12;
// ALL: @v12 {{.*}}, align 16
typedef __attribute__((ext_vector_type(248), aligned(4))) _Bool v12b_type;
v12b_type v12b;
// ALL: @v12b {{.*}}, align 4
