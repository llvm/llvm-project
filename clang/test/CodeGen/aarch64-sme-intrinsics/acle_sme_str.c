// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sme -target-feature +sve -S -O1 -Werror -emit-llvm -o - %s | FileCheck %s -check-prefixes=CHECK,CHECK-C
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sme -target-feature +sve -S -O1 -Werror -emit-llvm -o - -x c++ %s | FileCheck %s -check-prefixes=CHECK,CHECK-CXX
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sme -target-feature +sve -S -O1 -Werror -o /dev/null %s

#include <arm_sme_draft_spec_subject_to_change.h>

// CHECK-C-LABEL: @test_svstr_vnum_za(
// CHECK-CXX-LABEL: @_Z18test_svstr_vnum_zajPv(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.str(i32 [[SLICE_BASE:%.*]], ptr [[PTR:%.*]])
// CHECK-NEXT:    ret void
//
void test_svstr_vnum_za(uint32_t slice_base, void *ptr) {
  svstr_vnum_za(slice_base, ptr, 0);
}

// CHECK-C-LABEL: @test_svstr_vnum_za_1(
// CHECK-CXX-LABEL: @_Z20test_svstr_vnum_za_1jPv(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[SVLB:%.*]] = tail call i64 @llvm.aarch64.sme.cntsb()
// CHECK-NEXT:    [[MULVL:%.*]] = mul i64 [[SVLB]], 15
// CHECK-NEXT:    [[TMP0:%.*]] = getelementptr i8, ptr [[PTR:%.*]], i64 [[MULVL]]
// CHECK-NEXT:    [[TILESLICE:%.*]] = add i32 [[SLICE_BASE:%.*]], 15
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.str(i32 [[TILESLICE]], ptr [[TMP0]])
// CHECK-NEXT:    ret void
//
void test_svstr_vnum_za_1(uint32_t slice_base, void *ptr) {
  svstr_vnum_za(slice_base, ptr, 15);
}

// CHECK-C-LABEL: @test_svstr_za(
// CHECK-CXX-LABEL: @_Z13test_svstr_zajPv(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.str(i32 [[SLICE_BASE:%.*]], ptr [[PTR:%.*]])
// CHECK-NEXT:    ret void
//
void test_svstr_za(uint32_t slice_base, void *ptr) {
  svstr_za(slice_base, ptr);
}
