// RUN: %clang_cc1 -verify -fopenmp-simd -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s --implicit-check-not="{{__kmpc|__tgt|.floor_|.tile_}}"
// expected-no-diagnostics

// GH96383, GH183451: a loop transformation directive has no captured statement;
// in simd-only mode the original loop is emitted.

int a;

// CHECK-LABEL: define {{.*}}void @tile_global(
// CHECK:         store i32 0, ptr @a
// CHECK-NEXT:    br label %for.cond
// CHECK:       for.cond:
// CHECK-NEXT:    load i32, ptr @a
// CHECK-NEXT:    icmp slt i32 %{{.*}}, 2
// CHECK:       for.inc:
// CHECK-NEXT:    load i32, ptr @a
// CHECK-NEXT:    add nsw i32 %{{.*}}, 3
// CHECK-NEXT:    store i32 %{{.*}}, ptr @a
// CHECK-NEXT:    br label %for.cond
// CHECK:       for.end:
// CHECK-NEXT:    ret void
void tile_global(void) {
#pragma omp tile sizes(8)
  for (a = 0; a < 2; a = a + 3)
    ;
}
