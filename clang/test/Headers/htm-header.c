// RUN: %clang_cc1 -triple powerpc64le-unknown-unknown -target-feature +htm -DHTM_HEADER -ffreestanding -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64le-unknown-unknown -target-feature +htm -DHTM_HEADER -ffreestanding -emit-llvm -x c++ -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64le-unknown-unknown -target-feature +htm -DHTMXL_HEADER -ffreestanding -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64le-unknown-unknown -target-feature +htm -DHTMXL_HEADER -ffreestanding -emit-llvm -x c++ -o - %s | FileCheck %s

#ifdef HTM_HEADER
#include <htmintrin.h>
#endif

#ifdef HTMXL_HEADER
#include <htmxlintrin.h>
#endif

// Verify that simply including the headers does not generate any code
// (i.e. all inline routines in the header are marked "static")

// CHECK: target triple = "powerpc64
// CHECK-NEXT: {{^$}}
// CHECK-NEXT: {{llvm\..*}}
