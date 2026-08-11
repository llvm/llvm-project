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
// LLVM: call i16 @llvm.cttz.i16(i16 %{{.*}}, i1 false)
// LLVM: call i32 @llvm.cttz.i32(i32 %{{.*}}, i1 false)
// LLVM: call i64 @llvm.cttz.i64(i64 %{{.*}}, i1 false)
// LLVM: call i64 @llvm.cttz.i64(i64 %{{.*}}, i1 false)
// OGCG-LABEL: @test_stdc_trailing_zeros(
// OGCG: call i8 @llvm.cttz.i8(i8 %{{.*}}, i1 false)
// OGCG: call i16 @llvm.cttz.i16(i16 %{{.*}}, i1 false)
// OGCG: call i32 @llvm.cttz.i32(i32 %{{.*}}, i1 false)
// OGCG: call i64 @llvm.cttz.i64(i64 %{{.*}}, i1 false)
// OGCG: call i64 @llvm.cttz.i64(i64 %{{.*}}, i1 false)

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
// LLVM: call i16 @llvm.ctlz.i16(i16 %{{.*}}, i1 false)
// LLVM: call i32 @llvm.ctlz.i32(i32 %{{.*}}, i1 false)
// LLVM: call i64 @llvm.ctlz.i64(i64 %{{.*}}, i1 false)
// LLVM: call i64 @llvm.ctlz.i64(i64 %{{.*}}, i1 false)
// OGCG-LABEL: @test_stdc_leading_zeros(
// OGCG: call i8 @llvm.ctlz.i8(i8 %{{.*}}, i1 false)
// OGCG: call i16 @llvm.ctlz.i16(i16 %{{.*}}, i1 false)
// OGCG: call i32 @llvm.ctlz.i32(i32 %{{.*}}, i1 false)
// OGCG: call i64 @llvm.ctlz.i64(i64 %{{.*}}, i1 false)
// OGCG: call i64 @llvm.ctlz.i64(i64 %{{.*}}, i1 false)

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
// LLVM: call i16 @llvm.ctpop.i16(i16 %{{.*}})
// LLVM: call i32 @llvm.ctpop.i32(i32 %{{.*}})
// LLVM: call i64 @llvm.ctpop.i64(i64 %{{.*}})
// LLVM: call i64 @llvm.ctpop.i64(i64 %{{.*}})
// OGCG-LABEL: @test_stdc_count_ones(
// OGCG: call i8 @llvm.ctpop.i8(i8 %{{.*}})
// OGCG: call i16 @llvm.ctpop.i16(i16 %{{.*}})
// OGCG: call i32 @llvm.ctpop.i32(i32 %{{.*}})
// OGCG: call i64 @llvm.ctpop.i64(i64 %{{.*}})
// OGCG: call i64 @llvm.ctpop.i64(i64 %{{.*}})

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
// LLVM: call i16 @llvm.ctpop.i16(i16 %{{.*}})
// LLVM: sub i16 16, %{{.*}}
// LLVM: call i32 @llvm.ctpop.i32(i32 %{{.*}})
// LLVM: sub i32 32, %{{.*}}
// LLVM: call i64 @llvm.ctpop.i64(i64 %{{.*}})
// LLVM: sub i64 64, %{{.*}}
// LLVM: call i64 @llvm.ctpop.i64(i64 %{{.*}})
// LLVM: sub i64 64, %{{.*}}
// OGCG-LABEL: @test_stdc_count_zeros(
// OGCG: call i8 @llvm.ctpop.i8(i8 %{{.*}})
// OGCG: sub i8 8, %{{.*}}
// OGCG: call i16 @llvm.ctpop.i16(i16 %{{.*}})
// OGCG: sub i16 16, %{{.*}}
// OGCG: call i32 @llvm.ctpop.i32(i32 %{{.*}})
// OGCG: sub i32 32, %{{.*}}
// OGCG: call i64 @llvm.ctpop.i64(i64 %{{.*}})
// OGCG: sub i64 64, %{{.*}}
// OGCG: call i64 @llvm.ctpop.i64(i64 %{{.*}})
// OGCG: sub i64 64, %{{.*}}

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
// LLVM: call i16 @llvm.ctlz.i16(i16 %{{.*}}, i1 false)
// LLVM: sub i16 16, %{{.*}}
// LLVM: call i32 @llvm.ctlz.i32(i32 %{{.*}}, i1 false)
// LLVM: sub i32 32, %{{.*}}
// LLVM: call i64 @llvm.ctlz.i64(i64 %{{.*}}, i1 false)
// LLVM: sub i64 64, %{{.*}}
// LLVM: call i64 @llvm.ctlz.i64(i64 %{{.*}}, i1 false)
// LLVM: sub i64 64, %{{.*}}
// OGCG-LABEL: @test_stdc_bit_width(
// OGCG: call i8 @llvm.ctlz.i8(i8 %{{.*}}, i1 false)
// OGCG: sub i8 8, %{{.*}}
// OGCG: call i16 @llvm.ctlz.i16(i16 %{{.*}}, i1 false)
// OGCG: sub i16 16, %{{.*}}
// OGCG: call i32 @llvm.ctlz.i32(i32 %{{.*}}, i1 false)
// OGCG: sub i32 32, %{{.*}}
// OGCG: call i64 @llvm.ctlz.i64(i64 %{{.*}}, i1 false)
// OGCG: sub i64 64, %{{.*}}
// OGCG: call i64 @llvm.ctlz.i64(i64 %{{.*}}, i1 false)
// OGCG: sub i64 64, %{{.*}}

#else

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

#endif
