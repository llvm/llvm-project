// RUN: %clang_cc1 -triple powerpc64-unknown-unknown -target-feature +altivec -ffreestanding -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64-unknown-unknown -target-feature +altivec -ffreestanding -emit-llvm -fno-lax-vector-conversions -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64-unknown-unknown -target-feature +altivec -ffreestanding -emit-llvm -x c++ -o - %s | FileCheck %s

#include <altivec.h>

// Verify that simply including <altivec.h> does not generate any code
// (i.e. all inline routines in the header are marked "static")

// CHECK: target triple = "powerpc64-
// CHECK-NEXT: {{^$}}
// CHECK-NEXT: {{llvm\..*}}
