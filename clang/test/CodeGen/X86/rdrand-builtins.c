// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-unknown-unknown -target-feature +rdrnd -target-feature +rdseed -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK,X64
// RUN: %clang_cc1 -ffreestanding %s -triple=i386-unknown-unknown -target-feature +rdrnd -target-feature +rdseed -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK,X86

#include <immintrin.h>

int rdrand16(unsigned short *p) {
  return _rdrand16_step(p);
// CHECK: @rdrand16
// CHECK: call { i16, i32 } @llvm.x86.rdrand.16
// CHECK: store i16
}

int rdrand32(unsigned *p) {
  return _rdrand32_step(p);
// CHECK: @rdrand32
// CHECK: call { i32, i32 } @llvm.x86.rdrand.32
// CHECK: store i32
}

int rdrand64(unsigned long long *p) {
  return _rdrand64_step(p);
// X64: @rdrand64
// X64: call { i64, i32 } @llvm.x86.rdrand.64
// X64: store i64

// X86-LABEL: @rdrand64(
// X86-NEXT:  entry:
// X86-NEXT:    [[RETVAL_I:%.*]] = alloca i32, align 4
// X86-NEXT:    [[__P_ADDR_I:%.*]] = alloca ptr, align 4
// X86-NEXT:    [[__LO_I:%.*]] = alloca i32, align 4
// X86-NEXT:    [[__HI_I:%.*]] = alloca i32, align 4
// X86-NEXT:    [[__RES_LO_I:%.*]] = alloca i32, align 4
// X86-NEXT:    [[__RES_HI_I:%.*]] = alloca i32, align 4
// X86-NEXT:    [[P_ADDR:%.*]] = alloca ptr, align 4
// X86-NEXT:    store ptr [[P:%.*]], ptr [[P_ADDR]], align 4
// X86-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[P_ADDR]], align 4
// X86-NEXT:    store ptr [[TMP0]], ptr [[__P_ADDR_I]], align 4
// X86-NEXT:    [[TMP1:%.*]] = call { i32, i32 } @llvm.x86.rdrand.32()
// X86-NEXT:    [[TMP2:%.*]] = extractvalue { i32, i32 } [[TMP1]], 0
// X86-NEXT:    store i32 [[TMP2]], ptr [[__LO_I]], align 4
// X86-NEXT:    [[TMP3:%.*]] = extractvalue { i32, i32 } [[TMP1]], 1
// X86-NEXT:    store i32 [[TMP3]], ptr [[__RES_LO_I]], align 4
// X86-NEXT:    [[TMP4:%.*]] = call { i32, i32 } @llvm.x86.rdrand.32()
// X86-NEXT:    [[TMP5:%.*]] = extractvalue { i32, i32 } [[TMP4]], 0
// X86-NEXT:    store i32 [[TMP5]], ptr [[__HI_I]], align 4
// X86-NEXT:    [[TMP6:%.*]] = extractvalue { i32, i32 } [[TMP4]], 1
// X86-NEXT:    store i32 [[TMP6]], ptr [[__RES_HI_I]], align 4
// X86-NEXT:    [[TMP7:%.*]] = load i32, ptr [[__RES_LO_I]], align 4
// X86-NEXT:    [[TOBOOL_I:%.*]] = icmp ne i32 [[TMP7]], 0
// X86-NEXT:    br i1 [[TOBOOL_I]], label [[LAND_LHS_TRUE_I:%.*]], label [[IF_ELSE_I:%.*]]
// X86:       land.lhs.true.i:
// X86-NEXT:    [[TMP8:%.*]] = load i32, ptr [[__RES_HI_I]], align 4
// X86-NEXT:    [[TOBOOL1_I:%.*]] = icmp ne i32 [[TMP8]], 0
// X86-NEXT:    br i1 [[TOBOOL1_I]], label [[IF_THEN_I:%.*]], label [[IF_ELSE_I]]
// X86:       if.then.i:
// X86-NEXT:    [[TMP9:%.*]] = load i32, ptr [[__HI_I]], align 4
// X86-NEXT:    [[CONV_I:%.*]] = zext i32 [[TMP9]] to i64
// X86-NEXT:    [[SHL_I:%.*]] = shl i64 [[CONV_I]], 32
// X86-NEXT:    [[TMP10:%.*]] = load i32, ptr [[__LO_I]], align 4
// X86-NEXT:    [[CONV2_I:%.*]] = zext i32 [[TMP10]] to i64
// X86-NEXT:    [[OR_I:%.*]] = or i64 [[SHL_I]], [[CONV2_I]]
// X86-NEXT:    [[TMP11:%.*]] = load ptr, ptr [[__P_ADDR_I]], align 4
// X86-NEXT:    store i64 [[OR_I]], ptr [[TMP11]], align 4
// X86-NEXT:    store i32 1, ptr [[RETVAL_I]], align 4
// X86-NEXT:    br label [[_RDRAND64_STEP_EXIT:%.*]]
// X86:       if.else.i:
// X86-NEXT:    [[TMP12:%.*]] = load ptr, ptr [[__P_ADDR_I]], align 4
// X86-NEXT:    store i64 0, ptr [[TMP12]], align 4
// X86-NEXT:    store i32 0, ptr [[RETVAL_I]], align 4
// X86-NEXT:    br label [[_RDRAND64_STEP_EXIT]]
// X86:       _rdrand64_step.exit:
// X86-NEXT:    [[TMP13:%.*]] = load i32, ptr [[RETVAL_I]], align 4
// X86-NEXT:    ret i32 [[TMP13]]
}

int rdseed16(unsigned short *p) {
  return _rdseed16_step(p);
// CHECK: @rdseed16
// CHECK: call { i16, i32 } @llvm.x86.rdseed.16
// CHECK: store i16
}

int rdseed32(unsigned *p) {
  return _rdseed32_step(p);
// CHECK: @rdseed32
// CHECK: call { i32, i32 } @llvm.x86.rdseed.32
// CHECK: store i32
}

#if __x86_64__
int rdseed64(unsigned long long *p) {
  return _rdseed64_step(p);
// X64: @rdseed64
// X64: call { i64, i32 } @llvm.x86.rdseed.64
// X64: store i64
}
#endif
