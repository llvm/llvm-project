// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +v9.6a -O2 -emit-llvm -o - %s | FileCheck %s --check-prefix=IR
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +v9.6a -O2 -S -o - %s | FileCheck %s --check-prefix=ASM

#include <arm_acle.h>

// IR-LABEL: define dso_local void @test_release_store_ordering(
// IR:       if.then:
// IR-NEXT:    store i32 %v, ptr %b, align 4
// IR-NEXT:    %0 = zext i32 %v to i64
// IR-NEXT:    fence release
// IR-NEXT:    tail call void @llvm.aarch64.stshh.atomic.store.p0(ptr %a, i64 %0, i32 3, i32 0, i32 32)
// IR-NEXT:    br label %if.end
//
// ASM-LABEL: test_release_store_ordering:
// ASM:       str w2, [x1]
// ASM-NEXT:  mov w8, w2
// ASM-NEXT:  dmb ish
// ASM-NEXT:  stshh keep
// ASM-NEXT:  stlr w8, [x0]
void test_release_store_ordering(int *__restrict a, int *__restrict b, int v,
                                 int c) {
  if (c) {
    *b = v;
    __arm_atomic_store_with_stshh(a, v, __ATOMIC_RELEASE, 0);
  } else {
    *b = v + 1;
  }
}

// ASM-LABEL: demo_f32:
// ASM:       fmov w8, s0
// ASM-NEXT:  dmb ish
// ASM-NEXT:  stshh keep
// ASM-NEXT:  stlr w8, [x0]
void demo_f32(float *p, float v) {
  __arm_atomic_store_with_stshh(p, v, __ATOMIC_RELEASE, 0);
}
