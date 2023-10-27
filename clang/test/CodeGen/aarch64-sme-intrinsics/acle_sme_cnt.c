// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sme -target-feature +sve -S -O1 -Werror -emit-llvm -o - %s | FileCheck %s -check-prefixes=CHECK,CHECK-C
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sme -target-feature +sve -S -O1 -Werror -emit-llvm -o - -x c++ %s | FileCheck %s -check-prefixes=CHECK,CHECK-CXX
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sme -target-feature +sve -S -O1 -Werror -o /dev/null %s

#include <arm_sme_draft_spec_subject_to_change.h>

// CHECK-C-LABEL: @test_svcntsb(
// CHECK-CXX-LABEL: @_Z12test_svcntsbv(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call i64 @llvm.aarch64.sme.cntsb()
// CHECK-NEXT:    ret i64 [[TMP0]]
//
uint64_t test_svcntsb() {
  return svcntsb();
}

// CHECK-C-LABEL: @test_svcntsh(
// CHECK-CXX-LABEL: @_Z12test_svcntshv(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call i64 @llvm.aarch64.sme.cntsh()
// CHECK-NEXT:    ret i64 [[TMP0]]
//
uint64_t test_svcntsh() {
  return svcntsh();
}

// CHECK-C-LABEL: @test_svcntsw(
// CHECK-CXX-LABEL: @_Z12test_svcntswv(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call i64 @llvm.aarch64.sme.cntsw()
// CHECK-NEXT:    ret i64 [[TMP0]]
//
uint64_t test_svcntsw() {
  return svcntsw();
}

// CHECK-C-LABEL: @test_svcntsd(
// CHECK-CXX-LABEL: @_Z12test_svcntsdv(
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[TMP0:%.*]] = tail call i64 @llvm.aarch64.sme.cntsd()
// CHECK-NEXT:    ret i64 [[TMP0]]
//
uint64_t test_svcntsd() {
  return svcntsd();
}
