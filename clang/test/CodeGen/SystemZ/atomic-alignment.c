// RUN: %clang_cc1 -triple s390x-linux-gnu -O3 -emit-llvm %s -o - | FileCheck %s
//
// Test alignment of 128 bit Atomic int/fp types, as well as loading
// from memory with a simple addition. The fp128 is loaded as i128 and
// then casted.

// CHECK: @Atomic_int128 = {{.*}} i128 0, align 16
// CHECK: @Atomic_fp128 = {{.*}} fp128 0xL00000000000000000000000000000000, align 16

// CHECK-LABEL:  @f1
// CHECK:      %atomic-load = load atomic i128, ptr @Atomic_int128 seq_cst, align 16
// CHECK-NEXT: %add = add nsw i128 %atomic-load, 1
// CHECK-NEXT: store i128 %add, ptr %agg.result, align 8
// CHECK-NEXT: ret void

// CHECK-LABEL:  @f2
// CHECK:      %atomic-load = load atomic i128, ptr @Atomic_fp128 seq_cst, align 16
// CHECK-NEXT: %0 = bitcast i128 %atomic-load to fp128
// CHECK-NEXT: %add = fadd fp128 %0, 0xL00000000000000003FFF000000000000
// CHECK-NEXT: store fp128 %add, ptr %agg.result, align 8
// CHECK-NEXT: ret void


#include <stdatomic.h>

_Atomic __int128    Atomic_int128;
_Atomic long double Atomic_fp128;

__int128 f1() {
  return Atomic_int128 + 1;
}

long double f2() {
  return Atomic_fp128 + 1.0;
}
