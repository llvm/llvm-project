// RUN: %clang_cc1 -ffreestanding -triple x86_64-unknown-linux-gnu -std=c23 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -ffreestanding -triple x86_64-unknown-linux-gnu -std=c23 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -ffreestanding -triple x86_64-unknown-linux-gnu -std=c23 -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c23 -isystem %S/Inputs -DTEST_LIB_SPELLINGS -fclangir -emit-cir %s -o %t-lib.cir
// RUN: FileCheck --input-file=%t-lib.cir %s --check-prefix=LIB-CIR

#ifdef TEST_LIB_SPELLINGS
#include <stdbit.h>
#endif

#ifndef TEST_LIB_SPELLINGS

void test_stdc_trailing_zeros(unsigned char uc, unsigned short us,
                              unsigned int ui, unsigned long ul,
                              unsigned long long ull) {
  volatile unsigned int r;
  r = __builtin_stdc_trailing_zeros(uc);
  r = __builtin_stdc_trailing_zeros(us);
  r = __builtin_stdc_trailing_zeros(ui);
  r = __builtin_stdc_trailing_zeros(ul);
  r = __builtin_stdc_trailing_zeros(ull);
}

// CIR-LABEL: @test_stdc_trailing_zeros(
// CIR: cir.ctz %{{.+}} : !u8i
// CIR: cir.ctz %{{.+}} : !u16i
// CIR: cir.ctz %{{.+}} : !u32i
// CIR: cir.ctz %{{.+}} : !u64i
// CIR: cir.ctz %{{.+}} : !u64i
// LLVM-LABEL: @test_stdc_trailing_zeros(
// LLVM: call i8 @llvm.cttz.i8(i8 %{{.*}}, i1 false)
// LLVM: zext i8 %{{.*}} to i32
// LLVM: call i16 @llvm.cttz.i16(i16 %{{.*}}, i1 false)
// LLVM: zext i16 %{{.*}} to i32
// LLVM: call i32 @llvm.cttz.i32(i32 %{{.*}}, i1 false)
// LLVM: call i64 @llvm.cttz.i64(i64 %{{.*}}, i1 false)
// LLVM: trunc i64 %{{.*}} to i32
// LLVM: call i64 @llvm.cttz.i64(i64 %{{.*}}, i1 false)
// LLVM: trunc i64 %{{.*}} to i32
// OGCG-LABEL: @test_stdc_trailing_zeros(
// OGCG: call i8 @llvm.cttz.i8(i8 %{{.*}}, i1 false)
// OGCG: zext i8 %{{.*}} to i32
// OGCG: call i16 @llvm.cttz.i16(i16 %{{.*}}, i1 false)
// OGCG: zext i16 %{{.*}} to i32
// OGCG: call i32 @llvm.cttz.i32(i32 %{{.*}}, i1 false)
// OGCG: call i64 @llvm.cttz.i64(i64 %{{.*}}, i1 false)
// OGCG: trunc i64 %{{.*}} to i32
// OGCG: call i64 @llvm.cttz.i64(i64 %{{.*}}, i1 false)
// OGCG: trunc i64 %{{.*}} to i32

void test_stdc_trailing_ones(unsigned char uc, unsigned short us,
                             unsigned int ui, unsigned long ul,
                             unsigned long long ull) {
  volatile unsigned int r;
  r = __builtin_stdc_trailing_ones(uc);
  r = __builtin_stdc_trailing_ones(us);
  r = __builtin_stdc_trailing_ones(ui);
  r = __builtin_stdc_trailing_ones(ul);
  r = __builtin_stdc_trailing_ones(ull);
}

// CIR-LABEL: @test_stdc_trailing_ones(
// CIR: cir.not %{{.+}} : !u8i
// CIR: cir.ctz %{{.+}} : !u8i
// CIR: cir.not %{{.+}} : !u16i
// CIR: cir.ctz %{{.+}} : !u16i
// CIR: cir.not %{{.+}} : !u32i
// CIR: cir.ctz %{{.+}} : !u32i
// CIR: cir.not %{{.+}} : !u64i
// CIR: cir.ctz %{{.+}} : !u64i
// CIR: cir.not %{{.+}} : !u64i
// CIR: cir.ctz %{{.+}} : !u64i
// LLVM-LABEL: @test_stdc_trailing_ones(
// LLVM: xor i8 %{{.*}}, -1
// LLVM: call i8 @llvm.cttz.i8(i8 %{{.*}}, i1 false)
// LLVM: zext i8 %{{.*}} to i32
// LLVM: xor i16 %{{.*}}, -1
// LLVM: call i16 @llvm.cttz.i16(i16 %{{.*}}, i1 false)
// LLVM: zext i16 %{{.*}} to i32
// LLVM: xor i32 %{{.*}}, -1
// LLVM: call i32 @llvm.cttz.i32(i32 %{{.*}}, i1 false)
// LLVM: xor i64 %{{.*}}, -1
// LLVM: call i64 @llvm.cttz.i64(i64 %{{.*}}, i1 false)
// LLVM: trunc i64 %{{.*}} to i32
// LLVM: xor i64 %{{.*}}, -1
// LLVM: call i64 @llvm.cttz.i64(i64 %{{.*}}, i1 false)
// LLVM: trunc i64 %{{.*}} to i32
// OGCG-LABEL: @test_stdc_trailing_ones(
// OGCG: xor i8 %{{.*}}, -1
// OGCG: call i8 @llvm.cttz.i8(i8 %{{.*}}, i1 false)
// OGCG: zext i8 %{{.*}} to i32
// OGCG: xor i16 %{{.*}}, -1
// OGCG: call i16 @llvm.cttz.i16(i16 %{{.*}}, i1 false)
// OGCG: zext i16 %{{.*}} to i32
// OGCG: xor i32 %{{.*}}, -1
// OGCG: call i32 @llvm.cttz.i32(i32 %{{.*}}, i1 false)
// OGCG: xor i64 %{{.*}}, -1
// OGCG: call i64 @llvm.cttz.i64(i64 %{{.*}}, i1 false)
// OGCG: trunc i64 %{{.*}} to i32
// OGCG: xor i64 %{{.*}}, -1
// OGCG: call i64 @llvm.cttz.i64(i64 %{{.*}}, i1 false)
// OGCG: trunc i64 %{{.*}} to i32

unsigned int test_stdc_leading_zeros(unsigned char uc, unsigned short us,
                                     unsigned int ui, unsigned long ul,
                                     unsigned long long ull) {
  volatile unsigned int r;
  r = __builtin_stdc_leading_zeros(uc);
  r = __builtin_stdc_leading_zeros(us);
  r = __builtin_stdc_leading_zeros(ui);
  r = __builtin_stdc_leading_zeros(ul);
  r = __builtin_stdc_leading_zeros(ull);
  return r;
}

// CIR-LABEL: @test_stdc_leading_zeros(
// CIR: cir.clz %{{.+}} : !u8i
// CIR: cir.clz %{{.+}} : !u16i
// CIR: cir.clz %{{.+}} : !u32i
// CIR: cir.clz %{{.+}} : !u64i
// CIR: cir.clz %{{.+}} : !u64i
// LLVM-LABEL: @test_stdc_leading_zeros(
// LLVM: call i8 @llvm.ctlz.i8(i8 %{{.*}}, i1 false)
// LLVM: zext i8 %{{.*}} to i32
// LLVM: call i16 @llvm.ctlz.i16(i16 %{{.*}}, i1 false)
// LLVM: zext i16 %{{.*}} to i32
// LLVM: call i32 @llvm.ctlz.i32(i32 %{{.*}}, i1 false)
// LLVM: call i64 @llvm.ctlz.i64(i64 %{{.*}}, i1 false)
// LLVM: trunc i64 %{{.*}} to i32
// LLVM: call i64 @llvm.ctlz.i64(i64 %{{.*}}, i1 false)
// LLVM: trunc i64 %{{.*}} to i32
// OGCG-LABEL: @test_stdc_leading_zeros(
// OGCG: call i8 @llvm.ctlz.i8(i8 %{{.*}}, i1 false)
// OGCG: zext i8 %{{.*}} to i32
// OGCG: call i16 @llvm.ctlz.i16(i16 %{{.*}}, i1 false)
// OGCG: zext i16 %{{.*}} to i32
// OGCG: call i32 @llvm.ctlz.i32(i32 %{{.*}}, i1 false)
// OGCG: call i64 @llvm.ctlz.i64(i64 %{{.*}}, i1 false)
// OGCG: trunc i64 %{{.*}} to i32
// OGCG: call i64 @llvm.ctlz.i64(i64 %{{.*}}, i1 false)
// OGCG: trunc i64 %{{.*}} to i32

unsigned int test_stdc_leading_ones(unsigned char uc, unsigned short us,
                                    unsigned int ui, unsigned long ul,
                                    unsigned long long ull) {
  volatile unsigned int r;
  r = __builtin_stdc_leading_ones(uc);
  r = __builtin_stdc_leading_ones(us);
  r = __builtin_stdc_leading_ones(ui);
  r = __builtin_stdc_leading_ones(ul);
  r = __builtin_stdc_leading_ones(ull);
  return r;
}

// CIR-LABEL: @test_stdc_leading_ones(
// CIR: cir.not %{{.+}} : !u8i
// CIR: cir.clz %{{.+}} : !u8i
// CIR: cir.not %{{.+}} : !u16i
// CIR: cir.clz %{{.+}} : !u16i
// CIR: cir.not %{{.+}} : !u32i
// CIR: cir.clz %{{.+}} : !u32i
// CIR: cir.not %{{.+}} : !u64i
// CIR: cir.clz %{{.+}} : !u64i
// CIR: cir.not %{{.+}} : !u64i
// CIR: cir.clz %{{.+}} : !u64i
// LLVM-LABEL: @test_stdc_leading_ones(
// LLVM: xor i8 %{{.*}}, -1
// LLVM: call i8 @llvm.ctlz.i8(i8 %{{.*}}, i1 false)
// LLVM: zext i8 %{{.*}} to i32
// LLVM: xor i16 %{{.*}}, -1
// LLVM: call i16 @llvm.ctlz.i16(i16 %{{.*}}, i1 false)
// LLVM: zext i16 %{{.*}} to i32
// LLVM: xor i32 %{{.*}}, -1
// LLVM: call i32 @llvm.ctlz.i32(i32 %{{.*}}, i1 false)
// LLVM: xor i64 %{{.*}}, -1
// LLVM: call i64 @llvm.ctlz.i64(i64 %{{.*}}, i1 false)
// LLVM: trunc i64 %{{.*}} to i32
// LLVM: xor i64 %{{.*}}, -1
// LLVM: call i64 @llvm.ctlz.i64(i64 %{{.*}}, i1 false)
// LLVM: trunc i64 %{{.*}} to i32
// OGCG-LABEL: @test_stdc_leading_ones(
// OGCG: xor i8 %{{.*}}, -1
// OGCG: call i8 @llvm.ctlz.i8(i8 %{{.*}}, i1 false)
// OGCG: zext i8 %{{.*}} to i32
// OGCG: xor i16 %{{.*}}, -1
// OGCG: call i16 @llvm.ctlz.i16(i16 %{{.*}}, i1 false)
// OGCG: zext i16 %{{.*}} to i32
// OGCG: xor i32 %{{.*}}, -1
// OGCG: call i32 @llvm.ctlz.i32(i32 %{{.*}}, i1 false)
// OGCG: xor i64 %{{.*}}, -1
// OGCG: call i64 @llvm.ctlz.i64(i64 %{{.*}}, i1 false)
// OGCG: trunc i64 %{{.*}} to i32
// OGCG: xor i64 %{{.*}}, -1
// OGCG: call i64 @llvm.ctlz.i64(i64 %{{.*}}, i1 false)
// OGCG: trunc i64 %{{.*}} to i32

unsigned int test_stdc_count_ones(unsigned char uc, unsigned short us,
                                  unsigned int ui, unsigned long ul,
                                  unsigned long long ull) {
  volatile unsigned int r;
  r = __builtin_stdc_count_ones(uc);
  r = __builtin_stdc_count_ones(us);
  r = __builtin_stdc_count_ones(ui);
  r = __builtin_stdc_count_ones(ul);
  r = __builtin_stdc_count_ones(ull);
  return r;
}

// CIR-LABEL: @test_stdc_count_ones(
// CIR: cir.popcount %{{.+}} : !u8i
// CIR: cir.popcount %{{.+}} : !u16i
// CIR: cir.popcount %{{.+}} : !u32i
// CIR: cir.popcount %{{.+}} : !u64i
// CIR: cir.popcount %{{.+}} : !u64i
// LLVM-LABEL: @test_stdc_count_ones(
// LLVM: call i8 @llvm.ctpop.i8(i8 %{{.*}})
// LLVM: zext i8 %{{.*}} to i32
// LLVM: call i16 @llvm.ctpop.i16(i16 %{{.*}})
// LLVM: zext i16 %{{.*}} to i32
// LLVM: call i32 @llvm.ctpop.i32(i32 %{{.*}})
// LLVM: call i64 @llvm.ctpop.i64(i64 %{{.*}})
// LLVM: trunc i64 %{{.*}} to i32
// LLVM: call i64 @llvm.ctpop.i64(i64 %{{.*}})
// LLVM: trunc i64 %{{.*}} to i32
// OGCG-LABEL: @test_stdc_count_ones(
// OGCG: call i8 @llvm.ctpop.i8(i8 %{{.*}})
// OGCG: zext i8 %{{.*}} to i32
// OGCG: call i16 @llvm.ctpop.i16(i16 %{{.*}})
// OGCG: zext i16 %{{.*}} to i32
// OGCG: call i32 @llvm.ctpop.i32(i32 %{{.*}})
// OGCG: call i64 @llvm.ctpop.i64(i64 %{{.*}})
// OGCG: trunc i64 %{{.*}} to i32
// OGCG: call i64 @llvm.ctpop.i64(i64 %{{.*}})
// OGCG: trunc i64 %{{.*}} to i32

_Bool test_stdc_has_single_bit(unsigned char uc, unsigned short us,
                               unsigned int ui, unsigned long ul,
                               unsigned long long ull) {
  volatile _Bool r;
  r = __builtin_stdc_has_single_bit(uc);
  r = __builtin_stdc_has_single_bit(us);
  r = __builtin_stdc_has_single_bit(ui);
  r = __builtin_stdc_has_single_bit(ul);
  r = __builtin_stdc_has_single_bit(ull);
  return r;
}

// CIR-LABEL: @test_stdc_has_single_bit(
// CIR: %[[POPCOUNT_UC:.+]] = cir.popcount %{{.+}} : !u8i
// CIR: %[[ONE_UC:.+]] = cir.const #cir.int<1> : !u8i
// CIR: cir.cmp eq %[[POPCOUNT_UC]], %[[ONE_UC]] : !u8i
// CIR: %[[POPCOUNT_US:.+]] = cir.popcount %{{.+}} : !u16i
// CIR: %[[ONE_US:.+]] = cir.const #cir.int<1> : !u16i
// CIR: cir.cmp eq %[[POPCOUNT_US]], %[[ONE_US]] : !u16i
// CIR: %[[POPCOUNT_UI:.+]] = cir.popcount %{{.+}} : !u32i
// CIR: %[[ONE_UI:.+]] = cir.const #cir.int<1> : !u32i
// CIR: cir.cmp eq %[[POPCOUNT_UI]], %[[ONE_UI]] : !u32i
// CIR: %[[POPCOUNT_UL:.+]] = cir.popcount %{{.+}} : !u64i
// CIR: %[[ONE_UL:.+]] = cir.const #cir.int<1> : !u64i
// CIR: cir.cmp eq %[[POPCOUNT_UL]], %[[ONE_UL]] : !u64i
// CIR: %[[POPCOUNT_ULL:.+]] = cir.popcount %{{.+}} : !u64i
// CIR: %[[ONE_ULL:.+]] = cir.const #cir.int<1> : !u64i
// CIR: cir.cmp eq %[[POPCOUNT_ULL]], %[[ONE_ULL]] : !u64i
// LLVM-LABEL: @test_stdc_has_single_bit(
// LLVM: call i8 @llvm.ctpop.i8(i8 %{{.*}})
// LLVM: icmp eq i8 %{{.*}}, 1
// LLVM: call i16 @llvm.ctpop.i16(i16 %{{.*}})
// LLVM: icmp eq i16 %{{.*}}, 1
// LLVM: call i32 @llvm.ctpop.i32(i32 %{{.*}})
// LLVM: icmp eq i32 %{{.*}}, 1
// LLVM: call i64 @llvm.ctpop.i64(i64 %{{.*}})
// LLVM: icmp eq i64 %{{.*}}, 1
// LLVM: call i64 @llvm.ctpop.i64(i64 %{{.*}})
// LLVM: icmp eq i64 %{{.*}}, 1
// OGCG-LABEL: @test_stdc_has_single_bit(
// OGCG: call i8 @llvm.ctpop.i8(i8 %{{.*}})
// OGCG: icmp eq i8 %{{.*}}, 1
// OGCG: call i16 @llvm.ctpop.i16(i16 %{{.*}})
// OGCG: icmp eq i16 %{{.*}}, 1
// OGCG: call i32 @llvm.ctpop.i32(i32 %{{.*}})
// OGCG: icmp eq i32 %{{.*}}, 1
// OGCG: call i64 @llvm.ctpop.i64(i64 %{{.*}})
// OGCG: icmp eq i64 %{{.*}}, 1
// OGCG: call i64 @llvm.ctpop.i64(i64 %{{.*}})
// OGCG: icmp eq i64 %{{.*}}, 1

unsigned int test_stdc_count_zeros(unsigned char uc, unsigned short us,
                                   unsigned int ui, unsigned long ul,
                                   unsigned long long ull) {
  volatile unsigned int r;
  r = __builtin_stdc_count_zeros(uc);
  r = __builtin_stdc_count_zeros(us);
  r = __builtin_stdc_count_zeros(ui);
  r = __builtin_stdc_count_zeros(ul);
  r = __builtin_stdc_count_zeros(ull);
  return r;
}

// CIR-LABEL: @test_stdc_count_zeros(
// CIR: cir.popcount %{{.+}} : !u8i
// CIR: cir.sub %{{.+}}, %{{.+}} : !u8i
// CIR: cir.popcount %{{.+}} : !u16i
// CIR: cir.sub %{{.+}}, %{{.+}} : !u16i
// CIR: cir.popcount %{{.+}} : !u32i
// CIR: cir.sub %{{.+}}, %{{.+}} : !u32i
// CIR: cir.popcount %{{.+}} : !u64i
// CIR: cir.sub %{{.+}}, %{{.+}} : !u64i
// CIR: cir.popcount %{{.+}} : !u64i
// CIR: cir.sub %{{.+}}, %{{.+}} : !u64i
// LLVM-LABEL: @test_stdc_count_zeros(
// LLVM: call i8 @llvm.ctpop.i8(i8 %{{.*}})
// LLVM: sub i8 8, %{{.*}}
// LLVM: zext i8 %{{.*}} to i32
// LLVM: call i16 @llvm.ctpop.i16(i16 %{{.*}})
// LLVM: sub i16 16, %{{.*}}
// LLVM: zext i16 %{{.*}} to i32
// LLVM: call i32 @llvm.ctpop.i32(i32 %{{.*}})
// LLVM: sub i32 32, %{{.*}}
// LLVM: call i64 @llvm.ctpop.i64(i64 %{{.*}})
// LLVM: sub i64 64, %{{.*}}
// LLVM: trunc i64 %{{.*}} to i32
// LLVM: call i64 @llvm.ctpop.i64(i64 %{{.*}})
// LLVM: sub i64 64, %{{.*}}
// LLVM: trunc i64 %{{.*}} to i32
// OGCG-LABEL: @test_stdc_count_zeros(
// OGCG: call i8 @llvm.ctpop.i8(i8 %{{.*}})
// OGCG: sub i8 8, %{{.*}}
// OGCG: zext i8 %{{.*}} to i32
// OGCG: call i16 @llvm.ctpop.i16(i16 %{{.*}})
// OGCG: sub i16 16, %{{.*}}
// OGCG: zext i16 %{{.*}} to i32
// OGCG: call i32 @llvm.ctpop.i32(i32 %{{.*}})
// OGCG: sub i32 32, %{{.*}}
// OGCG: call i64 @llvm.ctpop.i64(i64 %{{.*}})
// OGCG: sub i64 64, %{{.*}}
// OGCG: trunc i64 %{{.*}} to i32
// OGCG: call i64 @llvm.ctpop.i64(i64 %{{.*}})
// OGCG: sub i64 64, %{{.*}}
// OGCG: trunc i64 %{{.*}} to i32

unsigned int test_stdc_bit_width(unsigned char uc, unsigned short us,
                                 unsigned int ui, unsigned long ul,
                                 unsigned long long ull) {
  volatile unsigned int r;
  r = __builtin_stdc_bit_width(uc);
  r = __builtin_stdc_bit_width(us);
  r = __builtin_stdc_bit_width(ui);
  r = __builtin_stdc_bit_width(ul);
  r = __builtin_stdc_bit_width(ull);
  return r;
}

// CIR-LABEL: @test_stdc_bit_width(
// CIR: cir.clz %{{.+}} : !u8i
// CIR: cir.sub %{{.+}}, %{{.+}} : !u8i
// CIR: cir.clz %{{.+}} : !u16i
// CIR: cir.sub %{{.+}}, %{{.+}} : !u16i
// CIR: cir.clz %{{.+}} : !u32i
// CIR: cir.sub %{{.+}}, %{{.+}} : !u32i
// CIR: cir.clz %{{.+}} : !u64i
// CIR: cir.sub %{{.+}}, %{{.+}} : !u64i
// CIR: cir.clz %{{.+}} : !u64i
// CIR: cir.sub %{{.+}}, %{{.+}} : !u64i
// LLVM-LABEL: @test_stdc_bit_width(
// LLVM: call i8 @llvm.ctlz.i8(i8 %{{.*}}, i1 false)
// LLVM: sub i8 8, %{{.*}}
// LLVM: zext i8 %{{.*}} to i32
// LLVM: call i16 @llvm.ctlz.i16(i16 %{{.*}}, i1 false)
// LLVM: sub i16 16, %{{.*}}
// LLVM: zext i16 %{{.*}} to i32
// LLVM: call i32 @llvm.ctlz.i32(i32 %{{.*}}, i1 false)
// LLVM: sub i32 32, %{{.*}}
// LLVM: call i64 @llvm.ctlz.i64(i64 %{{.*}}, i1 false)
// LLVM: sub i64 64, %{{.*}}
// LLVM: trunc i64 %{{.*}} to i32
// LLVM: call i64 @llvm.ctlz.i64(i64 %{{.*}}, i1 false)
// LLVM: sub i64 64, %{{.*}}
// LLVM: trunc i64 %{{.*}} to i32
// OGCG-LABEL: @test_stdc_bit_width(
// OGCG: call i8 @llvm.ctlz.i8(i8 %{{.*}}, i1 false)
// OGCG: sub i8 8, %{{.*}}
// OGCG: zext i8 %{{.*}} to i32
// OGCG: call i16 @llvm.ctlz.i16(i16 %{{.*}}, i1 false)
// OGCG: sub i16 16, %{{.*}}
// OGCG: zext i16 %{{.*}} to i32
// OGCG: call i32 @llvm.ctlz.i32(i32 %{{.*}}, i1 false)
// OGCG: sub i32 32, %{{.*}}
// OGCG: call i64 @llvm.ctlz.i64(i64 %{{.*}}, i1 false)
// OGCG: sub i64 64, %{{.*}}
// OGCG: trunc i64 %{{.*}} to i32
// OGCG: call i64 @llvm.ctlz.i64(i64 %{{.*}}, i1 false)
// OGCG: sub i64 64, %{{.*}}
// OGCG: trunc i64 %{{.*}} to i32

unsigned int test_stdc_first_leading_zero(unsigned char uc, unsigned short us,
                                          unsigned int ui, unsigned long ul,
                                          unsigned long long ull) {
  volatile unsigned int r;
  r = __builtin_stdc_first_leading_zero(uc);
  r = __builtin_stdc_first_leading_zero(us);
  r = __builtin_stdc_first_leading_zero(ui);
  r = __builtin_stdc_first_leading_zero(ul);
  r = __builtin_stdc_first_leading_zero(ull);
  return r;
}

// CIR-LABEL: @test_stdc_first_leading_zero(
// CIR: cir.not %{{.+}} : !u8i
// CIR: cir.clz %{{.+}} : !u8i
// CIR: cir.select
// CIR: cir.not %{{.+}} : !u16i
// CIR: cir.clz %{{.+}} : !u16i
// CIR: cir.select
// CIR: cir.not %{{.+}} : !u32i
// CIR: cir.clz %{{.+}} : !u32i
// CIR: cir.select
// CIR: cir.not %{{.+}} : !u64i
// CIR: cir.clz %{{.+}} : !u64i
// CIR: cir.select
// CIR: cir.not %{{.+}} : !u64i
// CIR: cir.clz %{{.+}} : !u64i
// CIR: cir.select
// LLVM-LABEL: @test_stdc_first_leading_zero(
// LLVM: xor i8 %{{.*}}, -1
// LLVM: call i8 @llvm.ctlz.i8(i8 %{{.*}}, i1 false)
// LLVM: zext i8 %{{.*}} to i32
// LLVM: xor i16 %{{.*}}, -1
// LLVM: call i16 @llvm.ctlz.i16(i16 %{{.*}}, i1 false)
// LLVM: zext i16 %{{.*}} to i32
// LLVM: xor i32 %{{.*}}, -1
// LLVM: call i32 @llvm.ctlz.i32(i32 %{{.*}}, i1 false)
// LLVM: xor i64 %{{.*}}, -1
// LLVM: call i64 @llvm.ctlz.i64(i64 %{{.*}}, i1 false)
// LLVM: trunc i64 %{{.*}} to i32
// LLVM: xor i64 %{{.*}}, -1
// LLVM: call i64 @llvm.ctlz.i64(i64 %{{.*}}, i1 false)
// LLVM: trunc i64 %{{.*}} to i32
// OGCG-LABEL: @test_stdc_first_leading_zero(
// OGCG: xor i8 %{{.*}}, -1
// OGCG: call i8 @llvm.ctlz.i8(i8 %{{.*}}, i1 false)
// OGCG: zext i8 %{{.*}} to i32
// OGCG: xor i16 %{{.*}}, -1
// OGCG: call i16 @llvm.ctlz.i16(i16 %{{.*}}, i1 false)
// OGCG: zext i16 %{{.*}} to i32
// OGCG: xor i32 %{{.*}}, -1
// OGCG: call i32 @llvm.ctlz.i32(i32 %{{.*}}, i1 false)
// OGCG: xor i64 %{{.*}}, -1
// OGCG: call i64 @llvm.ctlz.i64(i64 %{{.*}}, i1 false)
// OGCG: trunc i64 %{{.*}} to i32
// OGCG: xor i64 %{{.*}}, -1
// OGCG: call i64 @llvm.ctlz.i64(i64 %{{.*}}, i1 false)
// OGCG: trunc i64 %{{.*}} to i32

unsigned int test_stdc_first_leading_one(unsigned char uc, unsigned short us,
                                         unsigned int ui, unsigned long ul,
                                         unsigned long long ull) {
  volatile unsigned int r;
  r = __builtin_stdc_first_leading_one(uc);
  r = __builtin_stdc_first_leading_one(us);
  r = __builtin_stdc_first_leading_one(ui);
  r = __builtin_stdc_first_leading_one(ul);
  r = __builtin_stdc_first_leading_one(ull);
  return r;
}

// CIR-LABEL: @test_stdc_first_leading_one(
// CIR: cir.clz %{{.+}} : !u8i
// CIR: cir.select
// CIR: cir.clz %{{.+}} : !u16i
// CIR: cir.select
// CIR: cir.clz %{{.+}} : !u32i
// CIR: cir.select
// CIR: cir.clz %{{.+}} : !u64i
// CIR: cir.select
// CIR: cir.clz %{{.+}} : !u64i
// CIR: cir.select
// LLVM-LABEL: @test_stdc_first_leading_one(
// LLVM: call i8 @llvm.ctlz.i8(i8 %{{.*}}, i1 false)
// LLVM: zext i8 %{{.*}} to i32
// LLVM: call i16 @llvm.ctlz.i16(i16 %{{.*}}, i1 false)
// LLVM: zext i16 %{{.*}} to i32
// LLVM: call i32 @llvm.ctlz.i32(i32 %{{.*}}, i1 false)
// LLVM: call i64 @llvm.ctlz.i64(i64 %{{.*}}, i1 false)
// LLVM: trunc i64 %{{.*}} to i32
// LLVM: call i64 @llvm.ctlz.i64(i64 %{{.*}}, i1 false)
// LLVM: trunc i64 %{{.*}} to i32
// OGCG-LABEL: @test_stdc_first_leading_one(
// OGCG: call i8 @llvm.ctlz.i8(i8 %{{.*}}, i1 false)
// OGCG: zext i8 %{{.*}} to i32
// OGCG: call i16 @llvm.ctlz.i16(i16 %{{.*}}, i1 false)
// OGCG: zext i16 %{{.*}} to i32
// OGCG: call i32 @llvm.ctlz.i32(i32 %{{.*}}, i1 false)
// OGCG: call i64 @llvm.ctlz.i64(i64 %{{.*}}, i1 false)
// OGCG: trunc i64 %{{.*}} to i32
// OGCG: call i64 @llvm.ctlz.i64(i64 %{{.*}}, i1 false)
// OGCG: trunc i64 %{{.*}} to i32

unsigned int test_stdc_first_trailing_zero(unsigned char uc, unsigned short us,
                                           unsigned int ui, unsigned long ul,
                                           unsigned long long ull) {
  volatile unsigned int r;
  r = __builtin_stdc_first_trailing_zero(uc);
  r = __builtin_stdc_first_trailing_zero(us);
  r = __builtin_stdc_first_trailing_zero(ui);
  r = __builtin_stdc_first_trailing_zero(ul);
  r = __builtin_stdc_first_trailing_zero(ull);
  return r;
}

// CIR-LABEL: @test_stdc_first_trailing_zero(
// CIR: cir.not %{{.+}} : !u8i
// CIR: cir.ctz %{{.+}} : !u8i
// CIR: cir.select
// CIR: cir.not %{{.+}} : !u16i
// CIR: cir.ctz %{{.+}} : !u16i
// CIR: cir.select
// CIR: cir.not %{{.+}} : !u32i
// CIR: cir.ctz %{{.+}} : !u32i
// CIR: cir.select
// CIR: cir.not %{{.+}} : !u64i
// CIR: cir.ctz %{{.+}} : !u64i
// CIR: cir.select
// CIR: cir.not %{{.+}} : !u64i
// CIR: cir.ctz %{{.+}} : !u64i
// CIR: cir.select
// LLVM-LABEL: @test_stdc_first_trailing_zero(
// LLVM: xor i8 %{{.*}}, -1
// LLVM: call i8 @llvm.cttz.i8(i8 %{{.*}}, i1 false)
// LLVM: zext i8 %{{.*}} to i32
// LLVM: xor i16 %{{.*}}, -1
// LLVM: call i16 @llvm.cttz.i16(i16 %{{.*}}, i1 false)
// LLVM: zext i16 %{{.*}} to i32
// LLVM: xor i32 %{{.*}}, -1
// LLVM: call i32 @llvm.cttz.i32(i32 %{{.*}}, i1 false)
// LLVM: xor i64 %{{.*}}, -1
// LLVM: call i64 @llvm.cttz.i64(i64 %{{.*}}, i1 false)
// LLVM: trunc i64 %{{.*}} to i32
// LLVM: xor i64 %{{.*}}, -1
// LLVM: call i64 @llvm.cttz.i64(i64 %{{.*}}, i1 false)
// LLVM: trunc i64 %{{.*}} to i32
// OGCG-LABEL: @test_stdc_first_trailing_zero(
// OGCG: xor i8 %{{.*}}, -1
// OGCG: call i8 @llvm.cttz.i8(i8 %{{.*}}, i1 false)
// OGCG: zext i8 %{{.*}} to i32
// OGCG: xor i16 %{{.*}}, -1
// OGCG: call i16 @llvm.cttz.i16(i16 %{{.*}}, i1 false)
// OGCG: zext i16 %{{.*}} to i32
// OGCG: xor i32 %{{.*}}, -1
// OGCG: call i32 @llvm.cttz.i32(i32 %{{.*}}, i1 false)
// OGCG: xor i64 %{{.*}}, -1
// OGCG: call i64 @llvm.cttz.i64(i64 %{{.*}}, i1 false)
// OGCG: trunc i64 %{{.*}} to i32
// OGCG: xor i64 %{{.*}}, -1
// OGCG: call i64 @llvm.cttz.i64(i64 %{{.*}}, i1 false)
// OGCG: trunc i64 %{{.*}} to i32

unsigned int test_stdc_first_trailing_one(unsigned char uc, unsigned short us,
                                          unsigned int ui, unsigned long ul,
                                          unsigned long long ull) {
  volatile unsigned int r;
  r = __builtin_stdc_first_trailing_one(uc);
  r = __builtin_stdc_first_trailing_one(us);
  r = __builtin_stdc_first_trailing_one(ui);
  r = __builtin_stdc_first_trailing_one(ul);
  r = __builtin_stdc_first_trailing_one(ull);
  return r;
}

// CIR-LABEL: @test_stdc_first_trailing_one(
// CIR: cir.ctz %{{.+}} : !u8i
// CIR: cir.select
// CIR: cir.ctz %{{.+}} : !u16i
// CIR: cir.select
// CIR: cir.ctz %{{.+}} : !u32i
// CIR: cir.select
// CIR: cir.ctz %{{.+}} : !u64i
// CIR: cir.select
// CIR: cir.ctz %{{.+}} : !u64i
// CIR: cir.select
// LLVM-LABEL: @test_stdc_first_trailing_one(
// LLVM: call i8 @llvm.cttz.i8(i8 %{{.*}}, i1 false)
// LLVM: zext i8 %{{.*}} to i32
// LLVM: call i16 @llvm.cttz.i16(i16 %{{.*}}, i1 false)
// LLVM: zext i16 %{{.*}} to i32
// LLVM: call i32 @llvm.cttz.i32(i32 %{{.*}}, i1 false)
// LLVM: call i64 @llvm.cttz.i64(i64 %{{.*}}, i1 false)
// LLVM: trunc i64 %{{.*}} to i32
// LLVM: call i64 @llvm.cttz.i64(i64 %{{.*}}, i1 false)
// LLVM: trunc i64 %{{.*}} to i32
// OGCG-LABEL: @test_stdc_first_trailing_one(
// OGCG: call i8 @llvm.cttz.i8(i8 %{{.*}}, i1 false)
// OGCG: zext i8 %{{.*}} to i32
// OGCG: call i16 @llvm.cttz.i16(i16 %{{.*}}, i1 false)
// OGCG: zext i16 %{{.*}} to i32
// OGCG: call i32 @llvm.cttz.i32(i32 %{{.*}}, i1 false)
// OGCG: call i64 @llvm.cttz.i64(i64 %{{.*}}, i1 false)
// OGCG: trunc i64 %{{.*}} to i32
// OGCG: call i64 @llvm.cttz.i64(i64 %{{.*}}, i1 false)
// OGCG: trunc i64 %{{.*}} to i32

void test_stdc_bit_ceil(unsigned char uc, unsigned short us, unsigned int ui,
                        unsigned long ul, unsigned long long ull) {
  volatile unsigned char ruc;
  volatile unsigned short rus;
  volatile unsigned int rui;
  volatile unsigned long rul;
  volatile unsigned long long rull;
  ruc = __builtin_stdc_bit_ceil(uc);
  rus = __builtin_stdc_bit_ceil(us);
  rui = __builtin_stdc_bit_ceil(ui);
  rul = __builtin_stdc_bit_ceil(ul);
  rull = __builtin_stdc_bit_ceil(ull);
}

// CIR-LABEL: @test_stdc_bit_ceil(
// CIR: cir.sub %{{.+}}, %{{.+}} : !u8i
// CIR: cir.clz %{{.+}} : !u8i
// CIR: cir.sub %{{.+}}, %{{.+}} : !u8i
// CIR: cir.cmp le
// CIR: cir.shift(left, %{{.+}} : !u8i, %{{.+}} : !u8i) -> !u8i
// CIR: cir.shift(left, %{{.+}} : !u8i, %{{.+}} : !u8i) -> !u8i
// CIR: cir.select
// CIR: cir.sub %{{.+}}, %{{.+}} : !u16i
// CIR: cir.clz %{{.+}} : !u16i
// CIR: cir.sub %{{.+}}, %{{.+}} : !u16i
// CIR: cir.cmp le
// CIR: cir.shift(left, %{{.+}} : !u16i, %{{.+}} : !u16i) -> !u16i
// CIR: cir.shift(left, %{{.+}} : !u16i, %{{.+}} : !u16i) -> !u16i
// CIR: cir.select
// CIR: cir.sub %{{.+}}, %{{.+}} : !u32i
// CIR: cir.clz %{{.+}} : !u32i
// CIR: cir.sub %{{.+}}, %{{.+}} : !u32i
// CIR: cir.cmp le
// CIR: cir.shift(left, %{{.+}} : !u32i, %{{.+}} : !u32i) -> !u32i
// CIR: cir.shift(left, %{{.+}} : !u32i, %{{.+}} : !u32i) -> !u32i
// CIR: cir.select
// CIR: cir.sub %{{.+}}, %{{.+}} : !u64i
// CIR: cir.clz %{{.+}} : !u64i
// CIR: cir.sub %{{.+}}, %{{.+}} : !u64i
// CIR: cir.cmp le
// CIR: cir.shift(left, %{{.+}} : !u64i, %{{.+}} : !u64i) -> !u64i
// CIR: cir.shift(left, %{{.+}} : !u64i, %{{.+}} : !u64i) -> !u64i
// CIR: cir.select
// CIR: cir.sub %{{.+}}, %{{.+}} : !u64i
// CIR: cir.clz %{{.+}} : !u64i
// CIR: cir.sub %{{.+}}, %{{.+}} : !u64i
// CIR: cir.cmp le
// CIR: cir.shift(left, %{{.+}} : !u64i, %{{.+}} : !u64i) -> !u64i
// CIR: cir.shift(left, %{{.+}} : !u64i, %{{.+}} : !u64i) -> !u64i
// CIR: cir.select
// LLVM-LABEL: @test_stdc_bit_ceil(
// LLVM: call i8 @llvm.ctlz.i8(i8 %{{.*}}, i1 false)
// LLVM: sub i8 7, %{{.*}}
// LLVM: icmp ule i8 %{{.*}}, 1
// LLVM: shl i8 2, %{{.*}}
// LLVM: select i1 %{{.*}}, i8 1, i8 %{{.*}}
// LLVM: call i16 @llvm.ctlz.i16(i16 %{{.*}}, i1 false)
// LLVM: sub i16 15, %{{.*}}
// LLVM: icmp ule i16 %{{.*}}, 1
// LLVM: shl i16 2, %{{.*}}
// LLVM: select i1 %{{.*}}, i16 1, i16 %{{.*}}
// LLVM: call i32 @llvm.ctlz.i32(i32 %{{.*}}, i1 false)
// LLVM: sub i32 31, %{{.*}}
// LLVM: icmp ule i32 %{{.*}}, 1
// LLVM: shl i32 2, %{{.*}}
// LLVM: select i1 %{{.*}}, i32 1, i32 %{{.*}}
// LLVM: call i64 @llvm.ctlz.i64(i64 %{{.*}}, i1 false)
// LLVM: sub i64 63, %{{.*}}
// LLVM: icmp ule i64 %{{.*}}, 1
// LLVM: shl i64 2, %{{.*}}
// LLVM: select i1 %{{.*}}, i64 1, i64 %{{.*}}
// LLVM: call i64 @llvm.ctlz.i64(i64 %{{.*}}, i1 false)
// LLVM: sub i64 63, %{{.*}}
// LLVM: icmp ule i64 %{{.*}}, 1
// LLVM: shl i64 2, %{{.*}}
// LLVM: select i1 %{{.*}}, i64 1, i64 %{{.*}}
// OGCG-LABEL: @test_stdc_bit_ceil(
// OGCG: icmp ule i8 %{{.*}}, 1
// OGCG: call i8 @llvm.ctlz.i8(i8 %{{.*}}, i1 false)
// OGCG: sub i8 7, %{{.*}}
// OGCG: shl i8 2, %{{.*}}
// OGCG: icmp ule i16 %{{.*}}, 1
// OGCG: call i16 @llvm.ctlz.i16(i16 %{{.*}}, i1 false)
// OGCG: sub i16 15, %{{.*}}
// OGCG: shl i16 2, %{{.*}}
// OGCG: icmp ule i32 %{{.*}}, 1
// OGCG: call i32 @llvm.ctlz.i32(i32 %{{.*}}, i1 false)
// OGCG: sub i32 31, %{{.*}}
// OGCG: shl i32 2, %{{.*}}
// OGCG: icmp ule i64 %{{.*}}, 1
// OGCG: call i64 @llvm.ctlz.i64(i64 %{{.*}}, i1 false)
// OGCG: sub i64 63, %{{.*}}
// OGCG: shl i64 2, %{{.*}}
// OGCG: icmp ule i64 %{{.*}}, 1
// OGCG: call i64 @llvm.ctlz.i64(i64 %{{.*}}, i1 false)
// OGCG: sub i64 63, %{{.*}}
// OGCG: shl i64 2, %{{.*}}

void test_stdc_bit_floor(unsigned char uc, unsigned short us, unsigned int ui,
                         unsigned long ul, unsigned long long ull) {
  volatile unsigned char ruc;
  volatile unsigned short rus;
  volatile unsigned int rui;
  volatile unsigned long rul;
  volatile unsigned long long rull;
  ruc = __builtin_stdc_bit_floor(uc);
  rus = __builtin_stdc_bit_floor(us);
  rui = __builtin_stdc_bit_floor(ui);
  rul = __builtin_stdc_bit_floor(ul);
  rull = __builtin_stdc_bit_floor(ull);
}

// CIR-LABEL: @test_stdc_bit_floor(
// CIR: cir.clz %{{.+}} poison_zero : !u8i
// CIR: cir.sub %{{.+}}, %{{.+}} : !u8i
// CIR: cir.shift(left, %{{.+}} : !u8i, %{{.+}} : !u8i) -> !u8i
// CIR: cir.select
// CIR: cir.clz %{{.+}} poison_zero : !u16i
// CIR: cir.sub %{{.+}}, %{{.+}} : !u16i
// CIR: cir.shift(left, %{{.+}} : !u16i, %{{.+}} : !u16i) -> !u16i
// CIR: cir.select
// CIR: cir.clz %{{.+}} poison_zero : !u32i
// CIR: cir.sub %{{.+}}, %{{.+}} : !u32i
// CIR: cir.shift(left, %{{.+}} : !u32i, %{{.+}} : !u32i) -> !u32i
// CIR: cir.select
// CIR: cir.clz %{{.+}} poison_zero : !u64i
// CIR: cir.sub %{{.+}}, %{{.+}} : !u64i
// CIR: cir.shift(left, %{{.+}} : !u64i, %{{.+}} : !u64i) -> !u64i
// CIR: cir.select
// CIR: cir.clz %{{.+}} poison_zero : !u64i
// CIR: cir.sub %{{.+}}, %{{.+}} : !u64i
// CIR: cir.shift(left, %{{.+}} : !u64i, %{{.+}} : !u64i) -> !u64i
// CIR: cir.select
// LLVM-LABEL: @test_stdc_bit_floor(
// LLVM: call i8 @llvm.ctlz.i8(i8 %{{.*}}, i1 true)
// LLVM: sub i8 7, %{{.*}}
// LLVM: shl i8 1, %{{.*}}
// LLVM: select i1 %{{.*}}, i8 0, i8 %{{.*}}
// LLVM: call i16 @llvm.ctlz.i16(i16 %{{.*}}, i1 true)
// LLVM: sub i16 15, %{{.*}}
// LLVM: shl i16 1, %{{.*}}
// LLVM: select i1 %{{.*}}, i16 0, i16 %{{.*}}
// LLVM: call i32 @llvm.ctlz.i32(i32 %{{.*}}, i1 true)
// LLVM: sub i32 31, %{{.*}}
// LLVM: shl i32 1, %{{.*}}
// LLVM: select i1 %{{.*}}, i32 0, i32 %{{.*}}
// LLVM: call i64 @llvm.ctlz.i64(i64 %{{.*}}, i1 true)
// LLVM: sub i64 63, %{{.*}}
// LLVM: shl i64 1, %{{.*}}
// LLVM: select i1 %{{.*}}, i64 0, i64 %{{.*}}
// LLVM: call i64 @llvm.ctlz.i64(i64 %{{.*}}, i1 true)
// LLVM: sub i64 63, %{{.*}}
// LLVM: shl i64 1, %{{.*}}
// LLVM: select i1 %{{.*}}, i64 0, i64 %{{.*}}
// OGCG-LABEL: @test_stdc_bit_floor(
// OGCG: call i8 @llvm.ctlz.i8(i8 %{{.*}}, i1 true)
// OGCG: sub i8 7, %{{.*}}
// OGCG: shl i8 1, %{{.*}}
// OGCG: select i1 %{{.*}}, i8 0, i8 %{{.*}}
// OGCG: call i16 @llvm.ctlz.i16(i16 %{{.*}}, i1 true)
// OGCG: sub i16 15, %{{.*}}
// OGCG: shl i16 1, %{{.*}}
// OGCG: select i1 %{{.*}}, i16 0, i16 %{{.*}}
// OGCG: call i32 @llvm.ctlz.i32(i32 %{{.*}}, i1 true)
// OGCG: sub i32 31, %{{.*}}
// OGCG: shl i32 1, %{{.*}}
// OGCG: select i1 %{{.*}}, i32 0, i32 %{{.*}}
// OGCG: call i64 @llvm.ctlz.i64(i64 %{{.*}}, i1 true)
// OGCG: sub i64 63, %{{.*}}
// OGCG: shl i64 1, %{{.*}}
// OGCG: select i1 %{{.*}}, i64 0, i64 %{{.*}}
// OGCG: call i64 @llvm.ctlz.i64(i64 %{{.*}}, i1 true)
// OGCG: sub i64 63, %{{.*}}
// OGCG: shl i64 1, %{{.*}}
// OGCG: select i1 %{{.*}}, i64 0, i64 %{{.*}}
#else

unsigned int test_stdc_trailing_zeros_lib(unsigned char uc, unsigned short us,
                                          unsigned int ui, unsigned long ul,
                                          unsigned long long ull) {
  volatile unsigned int r;
  r = stdc_trailing_zeros_uc(uc);
  r = stdc_trailing_zeros_us(us);
  r = stdc_trailing_zeros_ui(ui);
  r = stdc_trailing_zeros_ul(ul);
  r = stdc_trailing_zeros_ull(ull);
  return r;
}

// LIB-CIR-LABEL: @test_stdc_trailing_zeros_lib(
// LIB-CIR: cir.ctz %{{.+}} : !u8i
// LIB-CIR: cir.ctz %{{.+}} : !u16i
// LIB-CIR: cir.ctz %{{.+}} : !u32i
// LIB-CIR: cir.ctz %{{.+}} : !u64i
// LIB-CIR: cir.ctz %{{.+}} : !u64i

unsigned int test_stdc_leading_zeros_lib(unsigned char uc, unsigned short us,
                                         unsigned int ui, unsigned long ul,
                                         unsigned long long ull) {
  volatile unsigned int r;
  r = stdc_leading_zeros_uc(uc);
  r = stdc_leading_zeros_us(us);
  r = stdc_leading_zeros_ui(ui);
  r = stdc_leading_zeros_ul(ul);
  r = stdc_leading_zeros_ull(ull);
  return r;
}

// LIB-CIR-LABEL: @test_stdc_leading_zeros_lib(
// LIB-CIR: cir.clz %{{.+}} : !u8i
// LIB-CIR: cir.clz %{{.+}} : !u16i
// LIB-CIR: cir.clz %{{.+}} : !u32i
// LIB-CIR: cir.clz %{{.+}} : !u64i
// LIB-CIR: cir.clz %{{.+}} : !u64i

unsigned int test_stdc_count_ones_lib(unsigned char uc, unsigned short us,
                                      unsigned int ui, unsigned long ul,
                                      unsigned long long ull) {
  volatile unsigned int r;
  r = stdc_count_ones_uc(uc);
  r = stdc_count_ones_us(us);
  r = stdc_count_ones_ui(ui);
  r = stdc_count_ones_ul(ul);
  r = stdc_count_ones_ull(ull);
  return r;
}

// LIB-CIR-LABEL: @test_stdc_count_ones_lib(
// LIB-CIR: cir.popcount %{{.+}} : !u8i
// LIB-CIR: cir.popcount %{{.+}} : !u16i
// LIB-CIR: cir.popcount %{{.+}} : !u32i
// LIB-CIR: cir.popcount %{{.+}} : !u64i
// LIB-CIR: cir.popcount %{{.+}} : !u64i

unsigned int test_stdc_trailing_ones_lib(unsigned char uc, unsigned short us,
                                         unsigned int ui, unsigned long ul,
                                         unsigned long long ull) {
  volatile unsigned int r;
  r = stdc_trailing_ones_uc(uc);
  r = stdc_trailing_ones_us(us);
  r = stdc_trailing_ones_ui(ui);
  r = stdc_trailing_ones_ul(ul);
  r = stdc_trailing_ones_ull(ull);
  return r;
}

// LIB-CIR-LABEL: @test_stdc_trailing_ones_lib(
// LIB-CIR: cir.not %{{.+}} : !u8i
// LIB-CIR: cir.ctz %{{.+}} : !u8i
// LIB-CIR: cir.not %{{.+}} : !u16i
// LIB-CIR: cir.ctz %{{.+}} : !u16i
// LIB-CIR: cir.not %{{.+}} : !u32i
// LIB-CIR: cir.ctz %{{.+}} : !u32i
// LIB-CIR: cir.not %{{.+}} : !u64i
// LIB-CIR: cir.ctz %{{.+}} : !u64i
// LIB-CIR: cir.not %{{.+}} : !u64i
// LIB-CIR: cir.ctz %{{.+}} : !u64i

unsigned int test_stdc_leading_ones_lib(unsigned char uc, unsigned short us,
                                        unsigned int ui, unsigned long ul,
                                        unsigned long long ull) {
  volatile unsigned int r;
  r = stdc_leading_ones_uc(uc);
  r = stdc_leading_ones_us(us);
  r = stdc_leading_ones_ui(ui);
  r = stdc_leading_ones_ul(ul);
  r = stdc_leading_ones_ull(ull);
  return r;
}

// LIB-CIR-LABEL: @test_stdc_leading_ones_lib(
// LIB-CIR: cir.not %{{.+}} : !u8i
// LIB-CIR: cir.clz %{{.+}} : !u8i
// LIB-CIR: cir.not %{{.+}} : !u16i
// LIB-CIR: cir.clz %{{.+}} : !u16i
// LIB-CIR: cir.not %{{.+}} : !u32i
// LIB-CIR: cir.clz %{{.+}} : !u32i
// LIB-CIR: cir.not %{{.+}} : !u64i
// LIB-CIR: cir.clz %{{.+}} : !u64i
// LIB-CIR: cir.not %{{.+}} : !u64i
// LIB-CIR: cir.clz %{{.+}} : !u64i

unsigned int test_stdc_first_leading_zero_lib(unsigned char uc,
                                              unsigned short us,
                                              unsigned int ui,
                                              unsigned long ul,
                                              unsigned long long ull) {
  volatile unsigned int r;
  r = stdc_first_leading_zero_uc(uc);
  r = stdc_first_leading_zero_us(us);
  r = stdc_first_leading_zero_ui(ui);
  r = stdc_first_leading_zero_ul(ul);
  r = stdc_first_leading_zero_ull(ull);
  return r;
}

// LIB-CIR-LABEL: @test_stdc_first_leading_zero_lib(
// LIB-CIR: cir.not %{{.+}} : !u8i
// LIB-CIR: cir.clz %{{.+}} : !u8i
// LIB-CIR: cir.select
// LIB-CIR: cir.not %{{.+}} : !u16i
// LIB-CIR: cir.clz %{{.+}} : !u16i
// LIB-CIR: cir.select
// LIB-CIR: cir.not %{{.+}} : !u32i
// LIB-CIR: cir.clz %{{.+}} : !u32i
// LIB-CIR: cir.select
// LIB-CIR: cir.not %{{.+}} : !u64i
// LIB-CIR: cir.clz %{{.+}} : !u64i
// LIB-CIR: cir.select
// LIB-CIR: cir.not %{{.+}} : !u64i
// LIB-CIR: cir.clz %{{.+}} : !u64i
// LIB-CIR: cir.select

unsigned int test_stdc_first_leading_one_lib(unsigned char uc,
                                             unsigned short us,
                                             unsigned int ui,
                                             unsigned long ul,
                                             unsigned long long ull) {
  volatile unsigned int r;
  r = stdc_first_leading_one_uc(uc);
  r = stdc_first_leading_one_us(us);
  r = stdc_first_leading_one_ui(ui);
  r = stdc_first_leading_one_ul(ul);
  r = stdc_first_leading_one_ull(ull);
  return r;
}

// LIB-CIR-LABEL: @test_stdc_first_leading_one_lib(
// LIB-CIR: cir.clz %{{.+}} : !u8i
// LIB-CIR: cir.select
// LIB-CIR: cir.clz %{{.+}} : !u16i
// LIB-CIR: cir.select
// LIB-CIR: cir.clz %{{.+}} : !u32i
// LIB-CIR: cir.select
// LIB-CIR: cir.clz %{{.+}} : !u64i
// LIB-CIR: cir.select
// LIB-CIR: cir.clz %{{.+}} : !u64i
// LIB-CIR: cir.select

unsigned int test_stdc_first_trailing_zero_lib(unsigned char uc,
                                               unsigned short us,
                                               unsigned int ui,
                                               unsigned long ul,
                                               unsigned long long ull) {
  volatile unsigned int r;
  r = stdc_first_trailing_zero_uc(uc);
  r = stdc_first_trailing_zero_us(us);
  r = stdc_first_trailing_zero_ui(ui);
  r = stdc_first_trailing_zero_ul(ul);
  r = stdc_first_trailing_zero_ull(ull);
  return r;
}

// LIB-CIR-LABEL: @test_stdc_first_trailing_zero_lib(
// LIB-CIR: cir.not %{{.+}} : !u8i
// LIB-CIR: cir.ctz %{{.+}} : !u8i
// LIB-CIR: cir.select
// LIB-CIR: cir.not %{{.+}} : !u16i
// LIB-CIR: cir.ctz %{{.+}} : !u16i
// LIB-CIR: cir.select
// LIB-CIR: cir.not %{{.+}} : !u32i
// LIB-CIR: cir.ctz %{{.+}} : !u32i
// LIB-CIR: cir.select
// LIB-CIR: cir.not %{{.+}} : !u64i
// LIB-CIR: cir.ctz %{{.+}} : !u64i
// LIB-CIR: cir.select
// LIB-CIR: cir.not %{{.+}} : !u64i
// LIB-CIR: cir.ctz %{{.+}} : !u64i
// LIB-CIR: cir.select

unsigned int test_stdc_first_trailing_one_lib(unsigned char uc,
                                              unsigned short us,
                                              unsigned int ui,
                                              unsigned long ul,
                                              unsigned long long ull) {
  volatile unsigned int r;
  r = stdc_first_trailing_one_uc(uc);
  r = stdc_first_trailing_one_us(us);
  r = stdc_first_trailing_one_ui(ui);
  r = stdc_first_trailing_one_ul(ul);
  r = stdc_first_trailing_one_ull(ull);
  return r;
}

// LIB-CIR-LABEL: @test_stdc_first_trailing_one_lib(
// LIB-CIR: cir.ctz %{{.+}} : !u8i
// LIB-CIR: cir.select
// LIB-CIR: cir.ctz %{{.+}} : !u16i
// LIB-CIR: cir.select
// LIB-CIR: cir.ctz %{{.+}} : !u32i
// LIB-CIR: cir.select
// LIB-CIR: cir.ctz %{{.+}} : !u64i
// LIB-CIR: cir.select
// LIB-CIR: cir.ctz %{{.+}} : !u64i
// LIB-CIR: cir.select

_Bool test_stdc_has_single_bit_lib(unsigned char uc, unsigned short us,
                                   unsigned int ui, unsigned long ul,
                                   unsigned long long ull) {
  volatile _Bool r;
  r = stdc_has_single_bit_uc(uc);
  r = stdc_has_single_bit_us(us);
  r = stdc_has_single_bit_ui(ui);
  r = stdc_has_single_bit_ul(ul);
  r = stdc_has_single_bit_ull(ull);
  return r;
}

// LIB-CIR-LABEL: @test_stdc_has_single_bit_lib(
// LIB-CIR: %[[LIB_POPCOUNT_UC:.+]] = cir.popcount %{{.+}} : !u8i
// LIB-CIR: %[[LIB_ONE_UC:.+]] = cir.const #cir.int<1> : !u8i
// LIB-CIR: cir.cmp eq %[[LIB_POPCOUNT_UC]], %[[LIB_ONE_UC]] : !u8i
// LIB-CIR: %[[LIB_POPCOUNT_US:.+]] = cir.popcount %{{.+}} : !u16i
// LIB-CIR: %[[LIB_ONE_US:.+]] = cir.const #cir.int<1> : !u16i
// LIB-CIR: cir.cmp eq %[[LIB_POPCOUNT_US]], %[[LIB_ONE_US]] : !u16i
// LIB-CIR: %[[LIB_POPCOUNT_UI:.+]] = cir.popcount %{{.+}} : !u32i
// LIB-CIR: %[[LIB_ONE_UI:.+]] = cir.const #cir.int<1> : !u32i
// LIB-CIR: cir.cmp eq %[[LIB_POPCOUNT_UI]], %[[LIB_ONE_UI]] : !u32i
// LIB-CIR: %[[LIB_POPCOUNT_UL:.+]] = cir.popcount %{{.+}} : !u64i
// LIB-CIR: %[[LIB_ONE_UL:.+]] = cir.const #cir.int<1> : !u64i
// LIB-CIR: cir.cmp eq %[[LIB_POPCOUNT_UL]], %[[LIB_ONE_UL]] : !u64i
// LIB-CIR: %[[LIB_POPCOUNT_ULL:.+]] = cir.popcount %{{.+}} : !u64i
// LIB-CIR: %[[LIB_ONE_ULL:.+]] = cir.const #cir.int<1> : !u64i
// LIB-CIR: cir.cmp eq %[[LIB_POPCOUNT_ULL]], %[[LIB_ONE_ULL]] : !u64i

unsigned int test_stdc_count_zeros_lib(unsigned char uc, unsigned short us,
                                       unsigned int ui, unsigned long ul,
                                       unsigned long long ull) {
  volatile unsigned int r;
  r = stdc_count_zeros_uc(uc);
  r = stdc_count_zeros_us(us);
  r = stdc_count_zeros_ui(ui);
  r = stdc_count_zeros_ul(ul);
  r = stdc_count_zeros_ull(ull);
  return r;
}

// LIB-CIR-LABEL: @test_stdc_count_zeros_lib(
// LIB-CIR: cir.popcount %{{.+}} : !u8i
// LIB-CIR: cir.sub %{{.+}}, %{{.+}} : !u8i
// LIB-CIR: cir.popcount %{{.+}} : !u16i
// LIB-CIR: cir.sub %{{.+}}, %{{.+}} : !u16i
// LIB-CIR: cir.popcount %{{.+}} : !u32i
// LIB-CIR: cir.sub %{{.+}}, %{{.+}} : !u32i
// LIB-CIR: cir.popcount %{{.+}} : !u64i
// LIB-CIR: cir.sub %{{.+}}, %{{.+}} : !u64i
// LIB-CIR: cir.popcount %{{.+}} : !u64i
// LIB-CIR: cir.sub %{{.+}}, %{{.+}} : !u64i

unsigned int test_stdc_bit_width_lib(unsigned char uc, unsigned short us,
                                     unsigned int ui, unsigned long ul,
                                     unsigned long long ull) {
  volatile unsigned int r;
  r = stdc_bit_width_uc(uc);
  r = stdc_bit_width_us(us);
  r = stdc_bit_width_ui(ui);
  r = stdc_bit_width_ul(ul);
  r = stdc_bit_width_ull(ull);
  return r;
}

// LIB-CIR-LABEL: @test_stdc_bit_width_lib(
// LIB-CIR: cir.clz %{{.+}} : !u8i
// LIB-CIR: cir.sub %{{.+}}, %{{.+}} : !u8i
// LIB-CIR: cir.clz %{{.+}} : !u16i
// LIB-CIR: cir.sub %{{.+}}, %{{.+}} : !u16i
// LIB-CIR: cir.clz %{{.+}} : !u32i
// LIB-CIR: cir.sub %{{.+}}, %{{.+}} : !u32i
// LIB-CIR: cir.clz %{{.+}} : !u64i
// LIB-CIR: cir.sub %{{.+}}, %{{.+}} : !u64i
// LIB-CIR: cir.clz %{{.+}} : !u64i
// LIB-CIR: cir.sub %{{.+}}, %{{.+}} : !u64i

void test_stdc_bit_ceil_lib(unsigned char uc, unsigned short us,
                            unsigned int ui, unsigned long ul,
                            unsigned long long ull) {
  volatile unsigned char ruc;
  volatile unsigned short rus;
  volatile unsigned int rui;
  volatile unsigned long rul;
  volatile unsigned long long rull;
  ruc = stdc_bit_ceil_uc(uc);
  rus = stdc_bit_ceil_us(us);
  rui = stdc_bit_ceil_ui(ui);
  rul = stdc_bit_ceil_ul(ul);
  rull = stdc_bit_ceil_ull(ull);
}

// LIB-CIR-LABEL: @test_stdc_bit_ceil_lib(
// LIB-CIR: cir.sub %{{.+}}, %{{.+}} : !u8i
// LIB-CIR: cir.clz %{{.+}} : !u8i
// LIB-CIR: cir.sub %{{.+}}, %{{.+}} : !u8i
// LIB-CIR: cir.shift(left, %{{.+}} : !u8i, %{{.+}} : !u8i) -> !u8i
// LIB-CIR: cir.shift(left, %{{.+}} : !u8i, %{{.+}} : !u8i) -> !u8i
// LIB-CIR: cir.select
// LIB-CIR: cir.sub %{{.+}}, %{{.+}} : !u16i
// LIB-CIR: cir.clz %{{.+}} : !u16i
// LIB-CIR: cir.sub %{{.+}}, %{{.+}} : !u16i
// LIB-CIR: cir.shift(left, %{{.+}} : !u16i, %{{.+}} : !u16i) -> !u16i
// LIB-CIR: cir.shift(left, %{{.+}} : !u16i, %{{.+}} : !u16i) -> !u16i
// LIB-CIR: cir.select
// LIB-CIR: cir.sub %{{.+}}, %{{.+}} : !u32i
// LIB-CIR: cir.clz %{{.+}} : !u32i
// LIB-CIR: cir.sub %{{.+}}, %{{.+}} : !u32i
// LIB-CIR: cir.shift(left, %{{.+}} : !u32i, %{{.+}} : !u32i) -> !u32i
// LIB-CIR: cir.shift(left, %{{.+}} : !u32i, %{{.+}} : !u32i) -> !u32i
// LIB-CIR: cir.select
// LIB-CIR: cir.sub %{{.+}}, %{{.+}} : !u64i
// LIB-CIR: cir.clz %{{.+}} : !u64i
// LIB-CIR: cir.sub %{{.+}}, %{{.+}} : !u64i
// LIB-CIR: cir.shift(left, %{{.+}} : !u64i, %{{.+}} : !u64i) -> !u64i
// LIB-CIR: cir.shift(left, %{{.+}} : !u64i, %{{.+}} : !u64i) -> !u64i
// LIB-CIR: cir.select
// LIB-CIR: cir.sub %{{.+}}, %{{.+}} : !u64i
// LIB-CIR: cir.clz %{{.+}} : !u64i
// LIB-CIR: cir.sub %{{.+}}, %{{.+}} : !u64i
// LIB-CIR: cir.shift(left, %{{.+}} : !u64i, %{{.+}} : !u64i) -> !u64i
// LIB-CIR: cir.shift(left, %{{.+}} : !u64i, %{{.+}} : !u64i) -> !u64i
// LIB-CIR: cir.select

void test_stdc_bit_floor_lib(unsigned char uc, unsigned short us,
                             unsigned int ui, unsigned long ul,
                             unsigned long long ull) {
  volatile unsigned char ruc;
  volatile unsigned short rus;
  volatile unsigned int rui;
  volatile unsigned long rul;
  volatile unsigned long long rull;
  ruc = stdc_bit_floor_uc(uc);
  rus = stdc_bit_floor_us(us);
  rui = stdc_bit_floor_ui(ui);
  rul = stdc_bit_floor_ul(ul);
  rull = stdc_bit_floor_ull(ull);
}

// LIB-CIR-LABEL: @test_stdc_bit_floor_lib(
// LIB-CIR: cir.clz %{{.+}} poison_zero : !u8i
// LIB-CIR: cir.sub %{{.+}}, %{{.+}} : !u8i
// LIB-CIR: cir.shift(left, %{{.+}} : !u8i, %{{.+}} : !u8i) -> !u8i
// LIB-CIR: cir.select
// LIB-CIR: cir.clz %{{.+}} poison_zero : !u16i
// LIB-CIR: cir.sub %{{.+}}, %{{.+}} : !u16i
// LIB-CIR: cir.shift(left, %{{.+}} : !u16i, %{{.+}} : !u16i) -> !u16i
// LIB-CIR: cir.select
// LIB-CIR: cir.clz %{{.+}} poison_zero : !u32i
// LIB-CIR: cir.sub %{{.+}}, %{{.+}} : !u32i
// LIB-CIR: cir.shift(left, %{{.+}} : !u32i, %{{.+}} : !u32i) -> !u32i
// LIB-CIR: cir.select
// LIB-CIR: cir.clz %{{.+}} poison_zero : !u64i
// LIB-CIR: cir.sub %{{.+}}, %{{.+}} : !u64i
// LIB-CIR: cir.shift(left, %{{.+}} : !u64i, %{{.+}} : !u64i) -> !u64i
// LIB-CIR: cir.select
// LIB-CIR: cir.clz %{{.+}} poison_zero : !u64i
// LIB-CIR: cir.sub %{{.+}}, %{{.+}} : !u64i
// LIB-CIR: cir.shift(left, %{{.+}} : !u64i, %{{.+}} : !u64i) -> !u64i
// LIB-CIR: cir.select
#endif
