// REQUIRES: ppc64-registered-target
// RUN: %clang_cc1 -triple powerpc64-unknown-unknown -maltivec -ffreestanding -S -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64-unknown-unknown -maltivec -ffreestanding -fno-lax-vector-conversions -S -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64-unknown-unknown -maltivec -ffreestanding -x c++ -S -o - %s | FileCheck %s

#include <altivec.h>

// Verify that simply including <altivec.h> does not generate any code
// (i.e. all inline routines in the header are marked "static")

// CHECK-NOT: .text

