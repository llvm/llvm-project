// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sme -target-feature +sve -S -O1 -Werror -emit-llvm -o - %s | FileCheck %s -check-prefixes=CHECK,CHECK-C
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sme -target-feature +sve -S -O1 -Werror -emit-llvm -o - -x c++ %s | FileCheck %s -check-prefixes=CHECK,CHECK-CXX
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sme -target-feature +sve -S -O1 -Werror -o /dev/null %s

#include <arm_sme_draft_spec_subject_to_change.h>

// CHECK-C-LABEL: @test_svzero_mask_za(
// CHECK-CXX-LABEL: @_Z19test_svzero_mask_zav(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.zero(i32 0)
// CHECK-NEXT:    ret void
//
void test_svzero_mask_za() {
  svzero_mask_za(0);
}

// CHECK-C-LABEL: @test_svzero_mask_za_1(
// CHECK-CXX-LABEL: @_Z21test_svzero_mask_za_1v(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.zero(i32 176)
// CHECK-NEXT:    ret void
//
void test_svzero_mask_za_1() {
  svzero_mask_za(176);
}

// CHECK-C-LABEL: @test_svzero_mask_za_2(
// CHECK-CXX-LABEL: @_Z21test_svzero_mask_za_2v(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.zero(i32 255)
// CHECK-NEXT:    ret void
//
void test_svzero_mask_za_2() {
  svzero_mask_za(255);
}

// CHECK-C-LABEL: @test_svzero_za(
// CHECK-CXX-LABEL: @_Z14test_svzero_zav(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    tail call void @llvm.aarch64.sme.zero(i32 255)
// CHECK-NEXT:    ret void
//
void test_svzero_za() {
  svzero_za();
}
