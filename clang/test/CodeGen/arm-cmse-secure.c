// RUN: %clang_cc1 -triple thumbv8m.base-unknown-unknown-eabi   -emit-llvm -mrelocation-model static -mcmse -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple thumbebv8m.base-unknown-unknown-eabi -emit-llvm -mrelocation-model static -mcmse -o - %s | FileCheck %s

#include <arm_cmse.h>

unsigned test_cmse_primitives(void *p) {
// CHECK: define {{.*}} i32 @test_cmse_primitives
  cmse_address_info_t tt_val, ttt_val;
  cmse_address_info_t tta_val, ttat_val;
  unsigned sum;

  tt_val = cmse_TT(p);
  ttt_val = cmse_TTT(p);
  tta_val = cmse_TTA(p);
  ttat_val = cmse_TTAT(p);
// CHECK: call i32 @llvm.arm.cmse.tt
// CHECK: call i32 @llvm.arm.cmse.ttt
// CHECK: call i32 @llvm.arm.cmse.tta
// CHECK: call i32 @llvm.arm.cmse.ttat

  sum = tt_val.value;
  sum += ttt_val.value;
  sum += tta_val.value;
  sum += ttat_val.value;

  sum += tt_val.flags.mpu_region;
  sum += tt_val.flags.sau_region;
  sum += tt_val.flags.mpu_region_valid;
  sum += tt_val.flags.sau_region_valid;
  sum += tt_val.flags.read_ok;
  sum += tt_val.flags.readwrite_ok;
  sum += tt_val.flags.nonsecure_read_ok;
  sum += tt_val.flags.nonsecure_readwrite_ok;
  sum += tt_val.flags.secure;
  sum += tt_val.flags.idau_region_valid;
  sum += tt_val.flags.idau_region;

  return sum;
}

void *test_address_range(void *p) {
// CHECK: define {{.*}} ptr @test_address_range
  return cmse_check_address_range(p, 128, CMSE_MPU_UNPRIV
                                        | CMSE_MPU_NONSECURE
                                        | CMSE_MPU_READWRITE);
// CHECK: call i32 @llvm.arm.cmse.tt
// CHECK: call i32 @llvm.arm.cmse.ttt
// CHECK: call i32 @llvm.arm.cmse.tta
// CHECK: call i32 @llvm.arm.cmse.ttat
}

typedef struct {
  int x, y, z;
} Point;

void test_pointed_object(void *p) {
// CHECK: define {{.*}} void @test_pointed_object
  Point *pt = (Point *)p;
  cmse_check_pointed_object(pt, CMSE_NONSECURE
                              | CMSE_MPU_READ
                              | CMSE_AU_NONSECURE);
// CHECK: call i32 @llvm.arm.cmse.tt
// CHECK: call i32 @llvm.arm.cmse.ttt
// CHECK: call i32 @llvm.arm.cmse.tta
// CHECK: call i32 @llvm.arm.cmse.ttat
}
