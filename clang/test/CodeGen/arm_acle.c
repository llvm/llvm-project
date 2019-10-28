// RUN: %clang_cc1 -ffreestanding -triple armv8-eabi -target-cpu cortex-a57 -O2  -fno-experimental-new-pass-manager -S -emit-llvm -o - %s | FileCheck %s -check-prefix=ARM -check-prefix=AArch32 -check-prefix=ARM-LEGACY -check-prefix=AArch32-LEGACY
// RUN: %clang_cc1 -ffreestanding -triple armv8-eabi -target-cpu cortex-a57 -O2  -fexperimental-new-pass-manager -S -emit-llvm -o - %s | FileCheck %s -check-prefix=ARM -check-prefix=AArch32 -check-prefix=ARM-NEWPM -check-prefix=AArch32-NEWPM
// RUN: %clang_cc1 -ffreestanding -triple aarch64-eabi -target-cpu cortex-a57 -target-feature +neon -target-feature +crc -target-feature +crypto -O2 -fno-experimental-new-pass-manager -S -emit-llvm -o - %s | FileCheck %s -check-prefix=ARM -check-prefix=AArch64 -check-prefix=ARM-LEGACY -check-prefix=AArch64-LEGACY
// RUN: %clang_cc1 -ffreestanding -triple aarch64-eabi -target-cpu cortex-a57 -target-feature +neon -target-feature +crc -target-feature +crypto -O2 -fexperimental-new-pass-manager -S -emit-llvm -o - %s | FileCheck %s -check-prefix=ARM -check-prefix=AArch64 -check-prefix=ARM-NEWPM -check-prefix=AArch64-NEWPM
// RUN: %clang_cc1 -ffreestanding -triple aarch64-eabi -target-cpu cortex-a57 -target-feature +v8.3a -O2 -fexperimental-new-pass-manager -S -emit-llvm -o - %s | FileCheck %s -check-prefix=AArch64-v8_3
// RUN: %clang_cc1 -ffreestanding -triple aarch64-eabi -target-cpu cortex-a57 -target-feature +v8.4a -O2 -fexperimental-new-pass-manager -S -emit-llvm -o - %s | FileCheck %s -check-prefix=AArch64-v8_3
// RUN: %clang_cc1 -ffreestanding -triple aarch64-eabi -target-cpu cortex-a57 -target-feature +v8.5a -O2 -fexperimental-new-pass-manager -S -emit-llvm -o - %s | FileCheck %s -check-prefix=AArch64-v8_3

// REQUIRES: rewrite

#include <arm_acle.h>

/* 8 SYNCHRONIZATION, BARRIER AND HINT INTRINSICS */
/* 8.3 Memory Barriers */
// ARM-LABEL: test_dmb
// AArch32: call void @llvm.arm.dmb(i32 1)
// AArch64: call void @llvm.aarch64.dmb(i32 1)
void test_dmb(void) {
  __dmb(1);
}

// ARM-LABEL: test_dsb
// AArch32: call void @llvm.arm.dsb(i32 2)
// AArch64: call void @llvm.aarch64.dsb(i32 2)
void test_dsb(void) {
  __dsb(2);
}

// ARM-LABEL: test_isb
// AArch32: call void @llvm.arm.isb(i32 3)
// AArch64: call void @llvm.aarch64.isb(i32 3)
void test_isb(void) {
  __isb(3);
}

/* 8.4 Hints */
// ARM-LABEL: test_yield
// AArch32: call void @llvm.arm.hint(i32 1)
// AArch64: call void @llvm.aarch64.hint(i32 1)
void test_yield(void) {
  __yield();
}

// ARM-LABEL: test_wfe
// AArch32: call void @llvm.arm.hint(i32 2)
// AArch64: call void @llvm.aarch64.hint(i32 2)
void test_wfe(void) {
  __wfe();
}

// ARM-LABEL: test_wfi
// AArch32: call void @llvm.arm.hint(i32 3)
// AArch64: call void @llvm.aarch64.hint(i32 3)
void test_wfi(void) {
  __wfi();
}

// ARM-LABEL: test_sev
// AArch32: call void @llvm.arm.hint(i32 4)
// AArch64: call void @llvm.aarch64.hint(i32 4)
void test_sev(void) {
  __sev();
}

// ARM-LABEL: test_sevl
// AArch32: call void @llvm.arm.hint(i32 5)
// AArch64: call void @llvm.aarch64.hint(i32 5)
void test_sevl(void) {
  __sevl();
}

#if __ARM_32BIT_STATE
// AArch32-LABEL: test_dbg
// AArch32: call void @llvm.arm.dbg(i32 0)
void test_dbg(void) {
  __dbg(0);
}
#endif

/* 8.5 Swap */
// ARM-LABEL: test_swp
// AArch32: call i32 @llvm.arm.ldrex
// AArch32: call i32 @llvm.arm.strex
// AArch64: call i64 @llvm.aarch64.ldxr
// AArch64: call i32 @llvm.aarch64.stxr
void test_swp(uint32_t x, volatile void *p) {
  __swp(x, p);
}

/* 8.6 Memory prefetch intrinsics */
/* 8.6.1 Data prefetch */
// ARM-LABEL: test_pld
// ARM: call void @llvm.prefetch.p0i8(i8* null, i32 0, i32 3, i32 1)
void test_pld() {
  __pld(0);
}

// ARM-LABEL: test_pldx
// AArch32: call void @llvm.prefetch.p0i8(i8* null, i32 1, i32 3, i32 1)
// AArch64: call void @llvm.prefetch.p0i8(i8* null, i32 1, i32 1, i32 1)
void test_pldx() {
  __pldx(1, 2, 0, 0);
}

/* 8.6.2 Instruction prefetch */
// ARM-LABEL: test_pli
// ARM: call void @llvm.prefetch.p0i8(i8* null, i32 0, i32 3, i32 0)
void test_pli() {
  __pli(0);
}

// ARM-LABEL: test_plix
// AArch32: call void @llvm.prefetch.p0i8(i8* null, i32 0, i32 3, i32 0)
// AArch64: call void @llvm.prefetch.p0i8(i8* null, i32 0, i32 1, i32 0)
void test_plix() {
  __plix(2, 0, 0);
}

/* 8.7 NOP */
// ARM-LABEL: test_nop
// AArch32: call void @llvm.arm.hint(i32 0)
// AArch64: call void @llvm.aarch64.hint(i32 0)
void test_nop(void) {
  __nop();
}

/* 9 DATA-PROCESSING INTRINSICS */

/* 9.2 Miscellaneous data-processing intrinsics */
// ARM-LABEL: test_ror
// ARM-LEGACY: lshr
// ARM-LEGACY: sub
// ARM-LEGACY: shl
// ARM-LEGACY: or
// ARM-NEWPM: call i32 @llvm.fshr.i32(i32 %x, i32 %x, i32 %y)
uint32_t test_ror(uint32_t x, uint32_t y) {
  return __ror(x, y);
}

// ARM-LABEL: test_rorl
// ARM-LEGACY: lshr
// ARM-LEGACY: sub
// ARM-LEGACY: shl
// ARM-LEGACY: or
// AArch32-NEWPM: call i32 @llvm.fshr.i32(i32 %x, i32 %x, i32 %y)
unsigned long test_rorl(unsigned long x, uint32_t y) {
  return __rorl(x, y);
}

// ARM-LABEL: test_rorll
// ARM: lshr
// ARM: sub
// ARM: shl
// ARM: or
uint64_t test_rorll(uint64_t x, uint32_t y) {
  return __rorll(x, y);
}

// ARM-LABEL: test_clz
// ARM: call i32 @llvm.ctlz.i32(i32 %t, i1 false)
uint32_t test_clz(uint32_t t) {
  return __clz(t);
}

// ARM-LABEL: test_clzl
// AArch32: call i32 @llvm.ctlz.i32(i32 %t, i1 false)
// AArch64: call i64 @llvm.ctlz.i64(i64 %t, i1 false)
long test_clzl(long t) {
  return __clzl(t);
}

// ARM-LABEL: test_clzll
// ARM: call i64 @llvm.ctlz.i64(i64 %t, i1 false)
uint64_t test_clzll(uint64_t t) {
  return __clzll(t);
}

// ARM-LABEL: test_cls
// ARM: call i32 @llvm.arm.cls(i32 %t)
unsigned test_cls(uint32_t t) {
  return __cls(t);
}

// ARM-LABEL: test_clsl
// AArch32: call i32 @llvm.arm.cls(i32 %t)
// AArch64: call i32 @llvm.arm.cls64(i64 %t)
unsigned test_clsl(unsigned long t) {
  return __clsl(t);
}
// ARM-LABEL: test_clsll
// ARM: call i32 @llvm.arm.cls64(i64 %t)
unsigned test_clsll(uint64_t t) {
  return __clsll(t);
}

// ARM-LABEL: test_rev
// ARM: call i32 @llvm.bswap.i32(i32 %t)
uint32_t test_rev(uint32_t t) {
  return __rev(t);
}

// ARM-LABEL: test_revl
// AArch32: call i32 @llvm.bswap.i32(i32 %t)
// AArch64: call i64 @llvm.bswap.i64(i64 %t)
long test_revl(long t) {
  return __revl(t);
}

// ARM-LABEL: test_revll
// ARM: call i64 @llvm.bswap.i64(i64 %t)
uint64_t test_revll(uint64_t t) {
  return __revll(t);
}

// ARM-LABEL: test_rev16
// ARM: llvm.bswap
// ARM-LEGACY: lshr {{.*}}, 16
// ARM-LEGACY: shl {{.*}}, 16
// ARM-LEGACY: or
// ARM-NEWPM: call i32 @llvm.fshl.i32(i32 %0, i32 %0, i32 16)
uint32_t test_rev16(uint32_t t) {
  return __rev16(t);
}

// ARM-LABEL: test_rev16l
// AArch32: llvm.bswap
// AArch32-LEGACY: lshr {{.*}}, 16
// AArch32-LEGACY: shl {{.*}}, 16
// AArch32-LEGACY: or
// AArch32-NEWPM: call i32 @llvm.fshl.i32(i32 %0, i32 %0, i32 16)
// AArch64: [[T1:%.*]] = lshr i64 [[IN:%.*]], 32
// AArch64: [[T2:%.*]] = trunc i64 [[T1]] to i32
// AArch64: [[T3:%.*]] = tail call i32 @llvm.bswap.i32(i32 [[T2]])
// AArch64-LEGACY: [[T4:%.*]] = lshr i32 [[T3]], 16
// AArch64-LEGACY: [[T5:%.*]] = shl i32 [[T3]], 16
// AArch64-LEGACY: [[T6:%.*]] = or i32 [[T5]], [[T4]]
// AArch64-NEWPM: [[T6:%.*]] = tail call i32 @llvm.fshl.i32(i32 [[T3]], i32 [[T3]], i32 16)
// AArch64: [[T7:%.*]] = zext i32 [[T6]] to i64
// AArch64: [[T8:%.*]] = shl nuw i64 [[T7]], 32
// AArch64: [[T9:%.*]] = trunc i64 [[IN]] to i32
// AArch64: [[T10:%.*]] = tail call i32 @llvm.bswap.i32(i32 [[T9]])
// AArch64-LEGACY: [[T11:%.*]] = lshr i32 [[T10]], 16
// AArch64-LEGACY: [[T12:%.*]] = shl i32 [[T10]], 16
// AArch64-LEGACY: [[T13:%.*]] = or i32 [[T12]], [[T11]]
// AArch64-NEWPM: [[T13:%.*]] = tail call i32 @llvm.fshl.i32(i32 [[T10]], i32 [[T10]], i32 16)
// AArch64: [[T14:%.*]] = zext i32 [[T13]] to i64
// AArch64: [[T15:%.*]] = or i64 [[T8]], [[T14]]
long test_rev16l(long t) {
  return __rev16l(t);
}

// ARM-LABEL: test_rev16ll
// ARM: [[T1:%.*]] = lshr i64 [[IN:%.*]], 32
// ARM: [[T2:%.*]] = trunc i64 [[T1]] to i32
// ARM: [[T3:%.*]] = tail call i32 @llvm.bswap.i32(i32 [[T2]])
// ARM-LEGACY: [[T4:%.*]] = lshr i32 [[T3]], 16
// ARM-LEGACY: [[T5:%.*]] = shl i32 [[T3]], 16
// ARM-LEGACY: [[T6:%.*]] = or i32 [[T5]], [[T4]]
// ARM-NEWPM: [[T6:%.*]] = tail call i32 @llvm.fshl.i32(i32 [[T3]], i32 [[T3]], i32 16)
// ARM: [[T7:%.*]] = zext i32 [[T6]] to i64
// ARM: [[T8:%.*]] = shl nuw i64 [[T7]], 32
// ARM: [[T9:%.*]] = trunc i64 [[IN]] to i32
// ARM: [[T10:%.*]] = tail call i32 @llvm.bswap.i32(i32 [[T9]])
// ARM-LEGACY: [[T11:%.*]] = lshr i32 [[T10]], 16
// ARM-LEGACY: [[T12:%.*]] = shl i32 [[T10]], 16
// ARM-LEGACY: [[T13:%.*]] = or i32 [[T12]], [[T11]]
// ARM-NEWPM: [[T13:%.*]] = tail call i32 @llvm.fshl.i32(i32 [[T10]], i32 [[T10]], i32 16)
// ARM: [[T14:%.*]] = zext i32 [[T13]] to i64
// ARM: [[T15:%.*]] = or i64 [[T8]], [[T14]]
uint64_t test_rev16ll(uint64_t t) {
  return __rev16ll(t);
}

// ARM-LABEL: test_revsh
// ARM: call i16 @llvm.bswap.i16(i16 %t)
int16_t test_revsh(int16_t t) {
  return __revsh(t);
}

// ARM-LABEL: test_rbit
// AArch32: call i32 @llvm.bitreverse.i32
// AArch64: call i32 @llvm.bitreverse.i32
uint32_t test_rbit(uint32_t t) {
  return __rbit(t);
}

// ARM-LABEL: test_rbitl
// AArch32: call i32 @llvm.bitreverse.i32
// AArch64: call i64 @llvm.bitreverse.i64
long test_rbitl(long t) {
  return __rbitl(t);
}

// ARM-LABEL: test_rbitll
// AArch32: call i32 @llvm.bitreverse.i32
// AArch32: call i32 @llvm.bitreverse.i32
// AArch64: call i64 @llvm.bitreverse.i64
uint64_t test_rbitll(uint64_t t) {
  return __rbitll(t);
}

/* 9.4 Saturating intrinsics */
#ifdef __ARM_FEATURE_SAT
/* 9.4.1 Width-specified saturation intrinsics */
// AArch32-LABEL: test_ssat
// AArch32: call i32 @llvm.arm.ssat(i32 %t, i32 1)
int32_t test_ssat(int32_t t) {
  return __ssat(t, 1);
}

// AArch32-LABEL: test_usat
// AArch32: call i32 @llvm.arm.usat(i32 %t, i32 2)
uint32_t test_usat(int32_t t) {
  return __usat(t, 2);
}
#endif

/* 9.4.2 Saturating addition and subtraction intrinsics */
#ifdef __ARM_FEATURE_DSP
// AArch32-LABEL: test_qadd
// AArch32: call i32 @llvm.arm.qadd(i32 %a, i32 %b)
int32_t test_qadd(int32_t a, int32_t b) {
  return __qadd(a, b);
}

// AArch32-LABEL: test_qsub
// AArch32: call i32 @llvm.arm.qsub(i32 %a, i32 %b)
int32_t test_qsub(int32_t a, int32_t b) {
  return __qsub(a, b);
}

extern int32_t f();
// AArch32-LABEL: test_qdbl
// AArch32: [[VAR:%[a-z0-9]+]] = {{.*}} call {{.*}} @f
// AArch32-NOT: call {{.*}} @f
// AArch32: call i32 @llvm.arm.qadd(i32 [[VAR]], i32 [[VAR]])
int32_t test_qdbl() {
  return __qdbl(f());
}
#endif

/*
 * 9.3 16-bit multiplications
 */
#if __ARM_FEATURE_DSP
// AArch32-LABEL: test_smulbb
// AArch32: call i32 @llvm.arm.smulbb
int32_t test_smulbb(int32_t a, int32_t b) {
  return __smulbb(a, b);
}
// AArch32-LABEL: test_smulbt
// AArch32: call i32 @llvm.arm.smulbt
int32_t test_smulbt(int32_t a, int32_t b) {
  return __smulbt(a, b);
}
// AArch32-LABEL: test_smultb
// AArch32: call i32 @llvm.arm.smultb
int32_t test_smultb(int32_t a, int32_t b) {
  return __smultb(a, b);
}
// AArch32-LABEL: test_smultt
// AArch32: call i32 @llvm.arm.smultt
int32_t test_smultt(int32_t a, int32_t b) {
  return __smultt(a, b);
}
// AArch32-LABEL: test_smulwb
// AArch32: call i32 @llvm.arm.smulwb
int32_t test_smulwb(int32_t a, int32_t b) {
  return __smulwb(a, b);
}
// AArch32-LABEL: test_smulwt
// AArch32: call i32 @llvm.arm.smulwt
int32_t test_smulwt(int32_t a, int32_t b) {
  return __smulwt(a, b);
}
#endif

/* 9.4.3 Accumultating multiplications */
#if __ARM_FEATURE_DSP
// AArch32-LABEL: test_smlabb
// AArch32: call i32 @llvm.arm.smlabb(i32 %a, i32 %b, i32 %c)
int32_t test_smlabb(int32_t a, int32_t b, int32_t c) {
  return __smlabb(a, b, c);
}
// AArch32-LABEL: test_smlabt
// AArch32: call i32 @llvm.arm.smlabt(i32 %a, i32 %b, i32 %c)
int32_t test_smlabt(int32_t a, int32_t b, int32_t c) {
  return __smlabt(a, b, c);
}
// AArch32-LABEL: test_smlatb
// AArch32: call i32 @llvm.arm.smlatb(i32 %a, i32 %b, i32 %c)
int32_t test_smlatb(int32_t a, int32_t b, int32_t c) {
  return __smlatb(a, b, c);
}
// AArch32-LABEL: test_smlatt
// AArch32: call i32 @llvm.arm.smlatt(i32 %a, i32 %b, i32 %c)
int32_t test_smlatt(int32_t a, int32_t b, int32_t c) {
  return __smlatt(a, b, c);
}
// AArch32-LABEL: test_smlawb
// AArch32: call i32 @llvm.arm.smlawb(i32 %a, i32 %b, i32 %c)
int32_t test_smlawb(int32_t a, int32_t b, int32_t c) {
  return __smlawb(a, b, c);
}
// AArch32-LABEL: test_smlawt
// AArch32: call i32 @llvm.arm.smlawt(i32 %a, i32 %b, i32 %c)
int32_t test_smlawt(int32_t a, int32_t b, int32_t c) {
  return __smlawt(a, b, c);
}
#endif

/* 9.5.4 Parallel 16-bit saturation */
#if __ARM_FEATURE_SIMD32
// AArch32-LABEL: test_ssat16
// AArch32: call i32 @llvm.arm.ssat16
int16x2_t test_ssat16(int16x2_t a) {
  return __ssat16(a, 15);
}
// AArch32-LABEL: test_usat16
// AArch32: call i32 @llvm.arm.usat16
uint16x2_t test_usat16(int16x2_t a) {
  return __usat16(a, 15);
}
#endif

/* 9.5.5 Packing and unpacking */
#if __ARM_FEATURE_SIMD32
// AArch32-LABEL: test_sxtab16
// AArch32: call i32 @llvm.arm.sxtab16
int16x2_t test_sxtab16(int16x2_t a, int8x4_t b) {
  return __sxtab16(a, b);
}
// AArch32-LABEL: test_sxtb16
// AArch32: call i32 @llvm.arm.sxtb16
int16x2_t test_sxtb16(int8x4_t a) {
  return __sxtb16(a);
}
// AArch32-LABEL: test_uxtab16
// AArch32: call i32 @llvm.arm.uxtab16
int16x2_t test_uxtab16(int16x2_t a, int8x4_t b) {
  return __uxtab16(a, b);
}
// AArch32-LABEL: test_uxtb16
// AArch32: call i32 @llvm.arm.uxtb16
int16x2_t test_uxtb16(int8x4_t a) {
  return __uxtb16(a);
}
#endif

/* 9.5.6 Parallel selection */
#if __ARM_FEATURE_SIMD32
// AArch32-LABEL: test_sel
// AArch32: call i32 @llvm.arm.sel
uint8x4_t test_sel(uint8x4_t a, uint8x4_t b) {
  return __sel(a, b);
}
#endif

/* 9.5.7 Parallel 8-bit addition and subtraction */
#if __ARM_FEATURE_SIMD32
// AArch32-LABEL: test_qadd8
// AArch32: call i32 @llvm.arm.qadd8
int16x2_t test_qadd8(int8x4_t a, int8x4_t b) {
  return __qadd8(a, b);
}
// AArch32-LABEL: test_qsub8
// AArch32: call i32 @llvm.arm.qsub8
int8x4_t test_qsub8(int8x4_t a, int8x4_t b) {
  return __qsub8(a, b);
}
// AArch32-LABEL: test_sadd8
// AArch32: call i32 @llvm.arm.sadd8
int8x4_t test_sadd8(int8x4_t a, int8x4_t b) {
  return __sadd8(a, b);
}
// AArch32-LABEL: test_shadd8
// AArch32: call i32 @llvm.arm.shadd8
int8x4_t test_shadd8(int8x4_t a, int8x4_t b) {
  return __shadd8(a, b);
}
// AArch32-LABEL: test_shsub8
// AArch32: call i32 @llvm.arm.shsub8
int8x4_t test_shsub8(int8x4_t a, int8x4_t b) {
  return __shsub8(a, b);
}
// AArch32-LABEL: test_ssub8
// AArch32: call i32 @llvm.arm.ssub8
int8x4_t test_ssub8(int8x4_t a, int8x4_t b) {
  return __ssub8(a, b);
}
// AArch32-LABEL: test_uadd8
// AArch32: call i32 @llvm.arm.uadd8
uint8x4_t test_uadd8(uint8x4_t a, uint8x4_t b) {
  return __uadd8(a, b);
}
// AArch32-LABEL: test_uhadd8
// AArch32: call i32 @llvm.arm.uhadd8
uint8x4_t test_uhadd8(uint8x4_t a, uint8x4_t b) {
  return __uhadd8(a, b);
}
// AArch32-LABEL: test_uhsub8
// AArch32: call i32 @llvm.arm.uhsub8
uint8x4_t test_uhsub8(uint8x4_t a, uint8x4_t b) {
  return __uhsub8(a, b);
}
// AArch32-LABEL: test_uqadd8
// AArch32: call i32 @llvm.arm.uqadd8
uint8x4_t test_uqadd8(uint8x4_t a, uint8x4_t b) {
  return __uqadd8(a, b);
}
// AArch32-LABEL: test_uqsub8
// AArch32: call i32 @llvm.arm.uqsub8
uint8x4_t test_uqsub8(uint8x4_t a, uint8x4_t b) {
  return __uqsub8(a, b);
}
// AArch32-LABEL: test_usub8
// AArch32: call i32 @llvm.arm.usub8
uint8x4_t test_usub8(uint8x4_t a, uint8x4_t b) {
  return __usub8(a, b);
}
#endif

/* 9.5.8 Sum of 8-bit absolute differences */
#if __ARM_FEATURE_SIMD32
// AArch32-LABEL: test_usad8
// AArch32: call i32 @llvm.arm.usad8
uint32_t test_usad8(uint8x4_t a, uint8x4_t b) {
  return __usad8(a, b);
}
// AArch32-LABEL: test_usada8
// AArch32: call i32 @llvm.arm.usada8
uint32_t test_usada8(uint8_t a, uint8_t b, uint8_t c) {
  return __usada8(a, b, c);
}
#endif

/* 9.5.9 Parallel 16-bit addition and subtraction */
#if __ARM_FEATURE_SIMD32
// AArch32-LABEL: test_qadd16
// AArch32: call i32 @llvm.arm.qadd16
int16x2_t test_qadd16(int16x2_t a, int16x2_t b) {
  return __qadd16(a, b);
}
// AArch32-LABEL: test_qasx
// AArch32: call i32 @llvm.arm.qasx
int16x2_t test_qasx(int16x2_t a, int16x2_t b) {
  return __qasx(a, b);
}
// AArch32-LABEL: test_qsax
// AArch32: call i32 @llvm.arm.qsax
int16x2_t test_qsax(int16x2_t a, int16x2_t b) {
  return __qsax(a, b);
}
// AArch32-LABEL: test_qsub16
// AArch32: call i32 @llvm.arm.qsub16
int16x2_t test_qsub16(int16x2_t a, int16x2_t b) {
  return __qsub16(a, b);
}
// AArch32-LABEL: test_sadd16
// AArch32: call i32 @llvm.arm.sadd16
int16x2_t test_sadd16(int16x2_t a, int16x2_t b) {
  return __sadd16(a, b);
}
// AArch32-LABEL: test_sasx
// AArch32: call i32 @llvm.arm.sasx
int16x2_t test_sasx(int16x2_t a, int16x2_t b) {
  return __sasx(a, b);
}
// AArch32-LABEL: test_shadd16
// AArch32: call i32 @llvm.arm.shadd16
int16x2_t test_shadd16(int16x2_t a, int16x2_t b) {
  return __shadd16(a, b);
}
// AArch32-LABEL: test_shasx
// AArch32: call i32 @llvm.arm.shasx
int16x2_t test_shasx(int16x2_t a, int16x2_t b) {
  return __shasx(a, b);
}
// AArch32-LABEL: test_shsax
// AArch32: call i32 @llvm.arm.shsax
int16x2_t test_shsax(int16x2_t a, int16x2_t b) {
  return __shsax(a, b);
}
// AArch32-LABEL: test_shsub16
// AArch32: call i32 @llvm.arm.shsub16
int16x2_t test_shsub16(int16x2_t a, int16x2_t b) {
  return __shsub16(a, b);
}
// AArch32-LABEL: test_ssax
// AArch32: call i32 @llvm.arm.ssax
int16x2_t test_ssax(int16x2_t a, int16x2_t b) {
  return __ssax(a, b);
}
// AArch32-LABEL: test_ssub16
// AArch32: call i32 @llvm.arm.ssub16
int16x2_t test_ssub16(int16x2_t a, int16x2_t b) {
  return __ssub16(a, b);
}
// AArch32-LABEL: test_uadd16
// AArch32: call i32 @llvm.arm.uadd16
uint16x2_t test_uadd16(uint16x2_t a, uint16x2_t b) {
  return __uadd16(a, b);
}
// AArch32-LABEL: test_uasx
// AArch32: call i32 @llvm.arm.uasx
uint16x2_t test_uasx(uint16x2_t a, uint16x2_t b) {
  return __uasx(a, b);
}
// AArch32-LABEL: test_uhadd16
// AArch32: call i32 @llvm.arm.uhadd16
uint16x2_t test_uhadd16(uint16x2_t a, uint16x2_t b) {
  return __uhadd16(a, b);
}
// AArch32-LABEL: test_uhasx
// AArch32: call i32 @llvm.arm.uhasx
uint16x2_t test_uhasx(uint16x2_t a, uint16x2_t b) {
  return __uhasx(a, b);
}
// AArch32-LABEL: test_uhsax
// AArch32: call i32 @llvm.arm.uhsax
uint16x2_t test_uhsax(uint16x2_t a, uint16x2_t b) {
  return __uhsax(a, b);
}
// AArch32-LABEL: test_uhsub16
// AArch32: call i32 @llvm.arm.uhsub16
uint16x2_t test_uhsub16(uint16x2_t a, uint16x2_t b) {
  return __uhsub16(a, b);
}
// AArch32-LABEL: test_uqadd16
// AArch32: call i32 @llvm.arm.uqadd16
uint16x2_t test_uqadd16(uint16x2_t a, uint16x2_t b) {
  return __uqadd16(a, b);
}
// AArch32-LABEL: test_uqasx
// AArch32: call i32 @llvm.arm.uqasx
uint16x2_t test_uqasx(uint16x2_t a, uint16x2_t b) {
  return __uqasx(a, b);
}
// AArch32-LABEL: test_uqsax
// AArch32: call i32 @llvm.arm.uqsax
uint16x2_t test_uqsax(uint16x2_t a, uint16x2_t b) {
  return __uqsax(a, b);
}
// AArch32-LABEL: test_uqsub16
// AArch32: call i32 @llvm.arm.uqsub16
uint16x2_t test_uqsub16(uint16x2_t a, uint16x2_t b) {
  return __uqsub16(a, b);
}
// AArch32-LABEL: test_usax
// AArch32: call i32 @llvm.arm.usax
uint16x2_t test_usax(uint16x2_t a, uint16x2_t b) {
  return __usax(a, b);
}
// AArch32-LABEL: test_usub16
// AArch32: call i32 @llvm.arm.usub16
uint16x2_t test_usub16(uint16x2_t a, uint16x2_t b) {
  return __usub16(a, b);
}
#endif

/* 9.5.10 Parallel 16-bit multiplications */
#if __ARM_FEATURE_SIMD32
// AArch32-LABEL: test_smlad
// AArch32: call i32 @llvm.arm.smlad
int32_t test_smlad(int16x2_t a, int16x2_t b, int32_t c) {
  return __smlad(a, b, c);
}
// AArch32-LABEL: test_smladx
// AArch32: call i32 @llvm.arm.smladx
int32_t test_smladx(int16x2_t a, int16x2_t b, int32_t c) {
  return __smladx(a, b, c);
}
// AArch32-LABEL: test_smlald
// AArch32: call i64 @llvm.arm.smlald
int64_t test_smlald(int16x2_t a, int16x2_t b, int64_t c) {
  return __smlald(a, b, c);
}
// AArch32-LABEL: test_smlaldx
// AArch32: call i64 @llvm.arm.smlaldx
int64_t test_smlaldx(int16x2_t a, int16x2_t b, int64_t c) {
  return __smlaldx(a, b, c);
}
// AArch32-LABEL: test_smlsd
// AArch32: call i32 @llvm.arm.smlsd
int32_t test_smlsd(int16x2_t a, int16x2_t b, int32_t c) {
  return __smlsd(a, b, c);
}
// AArch32-LABEL: test_smlsdx
// AArch32: call i32 @llvm.arm.smlsdx
int32_t test_smlsdx(int16x2_t a, int16x2_t b, int32_t c) {
  return __smlsdx(a, b, c);
}
// AArch32-LABEL: test_smlsld
// AArch32: call i64 @llvm.arm.smlsld
int64_t test_smlsld(int16x2_t a, int16x2_t b, int64_t c) {
  return __smlsld(a, b, c);
}
// AArch32-LABEL: test_smlsldx
// AArch32: call i64 @llvm.arm.smlsldx
int64_t test_smlsldx(int16x2_t a, int16x2_t b, int64_t c) {
  return __smlsldx(a, b, c);
}
// AArch32-LABEL: test_smuad
// AArch32: call i32 @llvm.arm.smuad
int32_t test_smuad(int16x2_t a, int16x2_t b) {
  return __smuad(a, b);
}
// AArch32-LABEL: test_smuadx
// AArch32: call i32 @llvm.arm.smuadx
int32_t test_smuadx(int16x2_t a, int16x2_t b) {
  return __smuadx(a, b);
}
// AArch32-LABEL: test_smusd
// AArch32: call i32 @llvm.arm.smusd
int32_t test_smusd(int16x2_t a, int16x2_t b) {
  return __smusd(a, b);
}
// AArch32-LABEL: test_smusdx
// AArch32: call i32 @llvm.arm.smusdx
int32_t test_smusdx(int16x2_t a, int16x2_t b) {
  return __smusdx(a, b);
}
#endif

/* 9.7 CRC32 intrinsics */
// ARM-LABEL: test_crc32b
// AArch32: call i32 @llvm.arm.crc32b
// AArch64: call i32 @llvm.aarch64.crc32b
uint32_t test_crc32b(uint32_t a, uint8_t b) {
  return __crc32b(a, b);
}

// ARM-LABEL: test_crc32h
// AArch32: call i32 @llvm.arm.crc32h
// AArch64: call i32 @llvm.aarch64.crc32h
uint32_t test_crc32h(uint32_t a, uint16_t b) {
  return __crc32h(a, b);
}

// ARM-LABEL: test_crc32w
// AArch32: call i32 @llvm.arm.crc32w
// AArch64: call i32 @llvm.aarch64.crc32w
uint32_t test_crc32w(uint32_t a, uint32_t b) {
  return __crc32w(a, b);
}

// ARM-LABEL: test_crc32d
// AArch32: call i32 @llvm.arm.crc32w
// AArch32: call i32 @llvm.arm.crc32w
// AArch64: call i32 @llvm.aarch64.crc32x
uint32_t test_crc32d(uint32_t a, uint64_t b) {
  return __crc32d(a, b);
}

// ARM-LABEL: test_crc32cb
// AArch32: call i32 @llvm.arm.crc32cb
// AArch64: call i32 @llvm.aarch64.crc32cb
uint32_t test_crc32cb(uint32_t a, uint8_t b) {
  return __crc32cb(a, b);
}

// ARM-LABEL: test_crc32ch
// AArch32: call i32 @llvm.arm.crc32ch
// AArch64: call i32 @llvm.aarch64.crc32ch
uint32_t test_crc32ch(uint32_t a, uint16_t b) {
  return __crc32ch(a, b);
}

// ARM-LABEL: test_crc32cw
// AArch32: call i32 @llvm.arm.crc32cw
// AArch64: call i32 @llvm.aarch64.crc32cw
uint32_t test_crc32cw(uint32_t a, uint32_t b) {
  return __crc32cw(a, b);
}

// ARM-LABEL: test_crc32cd
// AArch32: call i32 @llvm.arm.crc32cw
// AArch32: call i32 @llvm.arm.crc32cw
// AArch64: call i32 @llvm.aarch64.crc32cx
uint32_t test_crc32cd(uint32_t a, uint64_t b) {
  return __crc32cd(a, b);
}

/* 10.1 Special register intrinsics */
// ARM-LABEL: test_rsr
// AArch64: call i64 @llvm.read_register.i64(metadata ![[M0:[0-9]]])
// AArch32: call i32 @llvm.read_register.i32(metadata ![[M2:[0-9]]])
uint32_t test_rsr() {
#ifdef __ARM_32BIT_STATE
  return __arm_rsr("cp1:2:c3:c4:5");
#else
  return __arm_rsr("1:2:3:4:5");
#endif
}

// ARM-LABEL: test_rsr64
// AArch64: call i64 @llvm.read_register.i64(metadata ![[M0:[0-9]]])
// AArch32: call i64 @llvm.read_register.i64(metadata ![[M3:[0-9]]])
uint64_t test_rsr64() {
#ifdef __ARM_32BIT_STATE
  return __arm_rsr64("cp1:2:c3");
#else
  return __arm_rsr64("1:2:3:4:5");
#endif
}

// ARM-LABEL: test_rsrp
// AArch64: call i64 @llvm.read_register.i64(metadata ![[M1:[0-9]]])
// AArch32: call i32 @llvm.read_register.i32(metadata ![[M4:[0-9]]])
void *test_rsrp() {
  return __arm_rsrp("sysreg");
}

// ARM-LABEL: test_wsr
// AArch64: call void @llvm.write_register.i64(metadata ![[M0:[0-9]]], i64 %{{.*}})
// AArch32: call void @llvm.write_register.i32(metadata ![[M2:[0-9]]], i32 %{{.*}})
void test_wsr(uint32_t v) {
#ifdef __ARM_32BIT_STATE
  __arm_wsr("cp1:2:c3:c4:5", v);
#else
  __arm_wsr("1:2:3:4:5", v);
#endif
}

// ARM-LABEL: test_wsr64
// AArch64: call void @llvm.write_register.i64(metadata ![[M0:[0-9]]], i64 %{{.*}})
// AArch32: call void @llvm.write_register.i64(metadata ![[M3:[0-9]]], i64 %{{.*}})
void test_wsr64(uint64_t v) {
#ifdef __ARM_32BIT_STATE
  __arm_wsr64("cp1:2:c3", v);
#else
  __arm_wsr64("1:2:3:4:5", v);
#endif
}

// ARM-LABEL: test_wsrp
// AArch64: call void @llvm.write_register.i64(metadata ![[M1:[0-9]]], i64 %{{.*}})
// AArch32: call void @llvm.write_register.i32(metadata ![[M4:[0-9]]], i32 %{{.*}})
void test_wsrp(void *v) {
  __arm_wsrp("sysreg", v);
}

// ARM-LABEL: test_rsrf
// AArch64: call i64 @llvm.read_register.i64(metadata ![[M0:[0-9]]])
// AArch32: call i32 @llvm.read_register.i32(metadata ![[M2:[0-9]]])
// ARM-NOT: uitofp
// ARM: bitcast
float test_rsrf() {
#ifdef __ARM_32BIT_STATE
  return __arm_rsrf("cp1:2:c3:c4:5");
#else
  return __arm_rsrf("1:2:3:4:5");
#endif
}
// ARM-LABEL: test_rsrf64
// AArch64: call i64 @llvm.read_register.i64(metadata ![[M0:[0-9]]])
// AArch32: call i64 @llvm.read_register.i64(metadata ![[M3:[0-9]]])
// ARM-NOT: uitofp
// ARM: bitcast
double test_rsrf64() {
#ifdef __ARM_32BIT_STATE
  return __arm_rsrf64("cp1:2:c3");
#else
  return __arm_rsrf64("1:2:3:4:5");
#endif
}
// ARM-LABEL: test_wsrf
// ARM-NOT: fptoui
// ARM: bitcast
// AArch64: call void @llvm.write_register.i64(metadata ![[M0:[0-9]]], i64 %{{.*}})
// AArch32: call void @llvm.write_register.i32(metadata ![[M2:[0-9]]], i32 %{{.*}})
void test_wsrf(float v) {
#ifdef __ARM_32BIT_STATE
  __arm_wsrf("cp1:2:c3:c4:5", v);
#else
  __arm_wsrf("1:2:3:4:5", v);
#endif
}
// ARM-LABEL: test_wsrf64
// ARM-NOT: fptoui
// ARM: bitcast
// AArch64: call void @llvm.write_register.i64(metadata ![[M0:[0-9]]], i64 %{{.*}})
// AArch32: call void @llvm.write_register.i64(metadata ![[M3:[0-9]]], i64 %{{.*}})
void test_wsrf64(double v) {
#ifdef __ARM_32BIT_STATE
  __arm_wsrf64("cp1:2:c3", v);
#else
  __arm_wsrf64("1:2:3:4:5", v);
#endif
}

// AArch32: ![[M2]] = !{!"cp1:2:c3:c4:5"}
// AArch32: ![[M3]] = !{!"cp1:2:c3"}
// AArch32: ![[M4]] = !{!"sysreg"}

// AArch64: ![[M0]] = !{!"1:2:3:4:5"}
// AArch64: ![[M1]] = !{!"sysreg"}

// AArch64-v8_3-LABEL: @test_jcvt(
// AArch64-v8_3: call i32 @llvm.aarch64.fjcvtzs
#ifdef __ARM_64BIT_STATE
int32_t test_jcvt(double v) {
  return __jcvt(v);
}
#endif
