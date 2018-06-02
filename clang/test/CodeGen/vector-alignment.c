// RUN: %clang_cc1 -w -triple x86_64-apple-darwin10 \
// RUN:  -emit-llvm -o - %s | FileCheck %s --check-prefix=ALL --check-prefix=DARWIN_SSE
// RUN: %clang_cc1 -w -triple   i386-apple-darwin10 \
// RUN:  -emit-llvm -o - %s | FileCheck %s --check-prefix=ALL --check-prefix=DARWIN_SSE
// RUN: %clang_cc1 -w -triple x86_64-apple-darwin10 -target-feature +avx \
// RUN:  -emit-llvm -o - %s | FileCheck %s --check-prefix=ALL --check-prefix=DARWIN_AVX
// RUN: %clang_cc1 -w -triple   i386-apple-darwin10 -target-feature +avx \
// RUN:  -emit-llvm -o - %s | FileCheck %s --check-prefix=ALL --check-prefix=DARWIN_AVX
// RUN: %clang_cc1 -w -triple x86_64-apple-darwin10 -target-feature +avx512f \
// RUN:  -emit-llvm -o - %s | FileCheck %s --check-prefix=ALL --check-prefix=DARWIN_AVX512
// RUN: %clang_cc1 -w -triple   i386-apple-darwin10 -target-feature +avx512f \
// RUN:  -emit-llvm -o - %s | FileCheck %s --check-prefix=ALL --check-prefix=DARWIN_AVX512
// RUN: %clang_cc1 -w -triple armv7-apple-ios10 \
// RUN:  -emit-llvm -o - %s | FileCheck %s --check-prefix=ALL --check-prefix=DARWIN_ARM32
// RUN: %clang_cc1 -w -triple arm64-apple-ios10 \
// RUN:  -emit-llvm -o - %s | FileCheck %s --check-prefix=ALL --check-prefix=DARWIN_ARM64

// RUN: %clang_cc1 -w -triple x86_64-pc-linux \
// RUN:  -emit-llvm -o - %s | FileCheck %s --check-prefix=ALL --check-prefix=GENERIC
// RUN: %clang_cc1 -w -triple   i386-pc-linux \
// RUN:  -emit-llvm -o - %s | FileCheck %s --check-prefix=ALL --check-prefix=GENERIC
// RUN: %clang_cc1 -w -triple x86_64-pc-linux -target-feature +avx \
// RUN:  -emit-llvm -o - %s | FileCheck %s --check-prefix=ALL --check-prefix=GENERIC
// RUN: %clang_cc1 -w -triple   i386-pc-linux -target-feature +avx \
// RUN:  -emit-llvm -o - %s | FileCheck %s --check-prefix=ALL --check-prefix=GENERIC
// RUN: %clang_cc1 -w -triple x86_64-pc-linux -target-feature +avx512f \
// RUN:  -emit-llvm -o - %s | FileCheck %s --check-prefix=ALL --check-prefix=GENERIC
// RUN: %clang_cc1 -w -triple   i386-pc-linux -target-feature +avx512f \
// RUN:  -emit-llvm -o - %s | FileCheck %s --check-prefix=ALL --check-prefix=GENERIC

// rdar://11759609

// At or below target max alignment with no aligned attribute should align based
// on the size of vector.
double __attribute__((vector_size(16))) v1;
// DARWIN_SSE: @v1 {{.*}}, align 16
// DARWIN_AVX: @v1 {{.*}}, align 16
// DARWIN_AVX512: @v1 {{.*}}, align 16
// DARWIN_ARM32: @v1 {{.*}}, align 16
// DARWIN_ARM64: @v1 {{.*}}, align 16
// GENERIC: @v1 {{.*}}, align 16
double __attribute__((vector_size(32))) v2;
// DARWIN_SSE: @v2 {{.*}}, align 16
// DARWIN_AVX: @v2 {{.*}}, align 16
// DARWIN_AVX512: @v2 {{.*}}, align 16
// DARWIN_ARM32: @v2 {{.*}}, align 16
// DARWIN_ARM64: @v2 {{.*}}, align 16
// GENERIC: @v2 {{.*}}, align 32

// Alignment above target max alignment with no aligned attribute should align
// based on the target max.
double __attribute__((vector_size(64))) v3;
// DARWIN_SSE: @v3 {{.*}}, align 16
// DARWIN_AVX: @v3 {{.*}}, align 16
// DARWIN_AVX512: @v3 {{.*}}, align 16
// DARWIN_ARM32: @v3 {{.*}}, align 16
// DARWIN_ARM64: @v3 {{.*}}, align 16
// GENERIC: @v3 {{.*}}, align 64
double __attribute__((vector_size(1024))) v4;
// DARWIN_SSE: @v4 {{.*}}, align 16
// DARWIN_AVX: @v4 {{.*}}, align 16
// DARWIN_AVX512: @v4 {{.*}}, align 16
// DARWIN_ARM32: @v4 {{.*}}, align 16
// DARWIN_ARM64: @v4 {{.*}}, align 16
// GENERIC: @v4 {{.*}}, align 1024

// Aliged attribute should always override.
double __attribute__((vector_size(16), aligned(16))) v5;
// ALL: @v5 {{.*}}, align 16
double __attribute__((vector_size(16), aligned(64))) v6;
// ALL: @v6 {{.*}}, align 64
double __attribute__((vector_size(32), aligned(16))) v7;
// ALL: @v7 {{.*}}, align 16
double __attribute__((vector_size(32), aligned(64))) v8;
// ALL: @v8 {{.*}}, align 64

// Check non-power of 2 widths.
double __attribute__((vector_size(24))) v9;
// DARWIN_SSE: @v9 {{.*}}, align 16
// DARWIN_AVX: @v9 {{.*}}, align 16
// DARWIN_AVX512: @v9 {{.*}}, align 16
// DARWIN_ARM32: @v9 {{.*}}, align 16
// DARWIN_ARM64: @v9 {{.*}}, align 16
// GENERIC: @v9 {{.*}}, align 32
double __attribute__((vector_size(40))) v10;
// DARWIN_SSE: @v10 {{.*}}, align 16
// DARWIN_AVX: @v10 {{.*}}, align 16
// DARWIN_AVX512: @v10 {{.*}}, align 16
// DARWIN_ARM32: @v10 {{.*}}, align 16
// DARWIN_ARM64: @v10 {{.*}}, align 16
// GENERIC: @v10 {{.*}}, align 64

// Check non-power of 2 widths with aligned attribute.
double __attribute__((vector_size(24), aligned(64))) v11;
// ALL: @v11 {{.*}}, align 64
double __attribute__((vector_size(80), aligned(16))) v12;
// ALL: @v12 {{.*}}, align 16
