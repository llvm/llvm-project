// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -D__ARM_FEATURE_SVE -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -o - %s >/dev/null 2>%t
// RUN: FileCheck --check-prefix=ASM --allow-empty %s <%t

// If this check fails please read test/CodeGen/aarch64-sve-intrinsics/README for instructions on how to resolve it.
// ASM-NOT: warning
#include <arm_sve.h>

uint64_t test_svcntb()
{
  // CHECK-LABEL: test_svcntb
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.cntb(i32 31)
  // CHECK: ret i64 %[[INTRINSIC]]
  return svcntb();
}

uint64_t test_svcntb_pat()
{
  // CHECK-LABEL: test_svcntb_pat
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.cntb(i32 0)
  // CHECK: ret i64 %[[INTRINSIC]]
  return svcntb_pat(SV_POW2);
}

uint64_t test_svcntb_pat_1()
{
  // CHECK-LABEL: test_svcntb_pat_1
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.cntb(i32 1)
  // CHECK: ret i64 %[[INTRINSIC]]
  return svcntb_pat(SV_VL1);
}

uint64_t test_svcntb_pat_2()
{
  // CHECK-LABEL: test_svcntb_pat_2
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.cntb(i32 2)
  // CHECK: ret i64 %[[INTRINSIC]]
  return svcntb_pat(SV_VL2);
}

uint64_t test_svcntb_pat_3()
{
  // CHECK-LABEL: test_svcntb_pat_3
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.cntb(i32 3)
  // CHECK: ret i64 %[[INTRINSIC]]
  return svcntb_pat(SV_VL3);
}

uint64_t test_svcntb_pat_4()
{
  // CHECK-LABEL: test_svcntb_pat_4
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.cntb(i32 4)
  // CHECK: ret i64 %[[INTRINSIC]]
  return svcntb_pat(SV_VL4);
}

uint64_t test_svcntb_pat_5()
{
  // CHECK-LABEL: test_svcntb_pat_5
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.cntb(i32 5)
  // CHECK: ret i64 %[[INTRINSIC]]
  return svcntb_pat(SV_VL5);
}

uint64_t test_svcntb_pat_6()
{
  // CHECK-LABEL: test_svcntb_pat_6
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.cntb(i32 6)
  // CHECK: ret i64 %[[INTRINSIC]]
  return svcntb_pat(SV_VL6);
}

uint64_t test_svcntb_pat_7()
{
  // CHECK-LABEL: test_svcntb_pat_7
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.cntb(i32 7)
  // CHECK: ret i64 %[[INTRINSIC]]
  return svcntb_pat(SV_VL7);
}

uint64_t test_svcntb_pat_8()
{
  // CHECK-LABEL: test_svcntb_pat_8
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.cntb(i32 8)
  // CHECK: ret i64 %[[INTRINSIC]]
  return svcntb_pat(SV_VL8);
}

uint64_t test_svcntb_pat_9()
{
  // CHECK-LABEL: test_svcntb_pat_9
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.cntb(i32 9)
  // CHECK: ret i64 %[[INTRINSIC]]
  return svcntb_pat(SV_VL16);
}

uint64_t test_svcntb_pat_10()
{
  // CHECK-LABEL: test_svcntb_pat_10
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.cntb(i32 10)
  // CHECK: ret i64 %[[INTRINSIC]]
  return svcntb_pat(SV_VL32);
}

uint64_t test_svcntb_pat_11()
{
  // CHECK-LABEL: test_svcntb_pat_11
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.cntb(i32 11)
  // CHECK: ret i64 %[[INTRINSIC]]
  return svcntb_pat(SV_VL64);
}

uint64_t test_svcntb_pat_12()
{
  // CHECK-LABEL: test_svcntb_pat_12
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.cntb(i32 12)
  // CHECK: ret i64 %[[INTRINSIC]]
  return svcntb_pat(SV_VL128);
}

uint64_t test_svcntb_pat_13()
{
  // CHECK-LABEL: test_svcntb_pat_13
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.cntb(i32 13)
  // CHECK: ret i64 %[[INTRINSIC]]
  return svcntb_pat(SV_VL256);
}

uint64_t test_svcntb_pat_14()
{
  // CHECK-LABEL: test_svcntb_pat_14
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.cntb(i32 29)
  // CHECK: ret i64 %[[INTRINSIC]]
  return svcntb_pat(SV_MUL4);
}

uint64_t test_svcntb_pat_15()
{
  // CHECK-LABEL: test_svcntb_pat_15
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.cntb(i32 30)
  // CHECK: ret i64 %[[INTRINSIC]]
  return svcntb_pat(SV_MUL3);
}

uint64_t test_svcntb_pat_16()
{
  // CHECK-LABEL: test_svcntb_pat_16
  // CHECK: %[[INTRINSIC:.*]] = call i64 @llvm.aarch64.sve.cntb(i32 31)
  // CHECK: ret i64 %[[INTRINSIC]]
  return svcntb_pat(SV_ALL);
}
