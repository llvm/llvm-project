// REQUIRES: powerpc-registered-target

// RUN: %clang -S -emit-llvm -target powerpc64-unknown-linux-gnu -mcpu=pwr8 -ffreestanding -DNO_WARN_X86_INTRINSICS %s \
// RUN: -ffp-contract=off -fno-discard-value-names -mllvm -disable-llvm-optzns -o - | llvm-cxxfilt -n | FileCheck %s --check-prefixes=CHECK,CHECK-BE
// RUN: %clang -x c++ -fsyntax-only -target powerpc64-unknown-linux-gnu -mcpu=pwr8 -ffreestanding -DNO_WARN_X86_INTRINSICS %s \
// RUN:   -fno-discard-value-names -mllvm -disable-llvm-optzns
// RUN: %clang -S -emit-llvm -target powerpc64le-unknown-linux-gnu -mcpu=pwr8 -ffreestanding -DNO_WARN_X86_INTRINSICS %s \
// RUN: -ffp-contract=off -fno-discard-value-names -mllvm -disable-llvm-optzns -o - | llvm-cxxfilt -n | FileCheck %s --check-prefixes=CHECK,CHECK-LE
// RUN: %clang -x c++ -fsyntax-only -target powerpc64le-unknown-linux-gnu -mcpu=pwr8 -ffreestanding -DNO_WARN_X86_INTRINSICS %s \
// RUN:   -fno-discard-value-names -mllvm -disable-llvm-optzns

// RUN: %clang -S -emit-llvm -target powerpc64le-unknown-linux-gnu -mcpu=pwr10 -ffreestanding -DNO_WARN_X86_INTRINSICS %s \
// RUN:   -ffp-contract=off -fno-discard-value-names -mllvm -disable-llvm-optzns -o - | llvm-cxxfilt -n | FileCheck %s --check-prefixes=CHECK,CHECK-P10-LE

// RUN: %clang -S -emit-llvm -target powerpc64-unknown-freebsd13.0 -mcpu=pwr8 -ffreestanding -nostdlibinc -DNO_WARN_X86_INTRINSICS %s \
// RUN:   -fno-discard-value-names -mllvm -disable-llvm-optzns -o - | llvm-cxxfilt -n | FileCheck %s --check-prefixes=CHECK,CHECK-BE
// RUN: %clang -x c++ -fsyntax-only -target powerpc64-unknown-freebsd13.0 -mcpu=pwr8 -ffreestanding -nostdlibinc -DNO_WARN_X86_INTRINSICS %s \
// RUN:   -fno-discard-value-names -mllvm -disable-llvm-optzns
// RUN: %clang -S -emit-llvm -target powerpc64le-unknown-freebsd13.0 -mcpu=pwr8 -ffreestanding -nostdlibinc -DNO_WARN_X86_INTRINSICS %s \
// RUN:   -fno-discard-value-names -mllvm -disable-llvm-optzns -o - | llvm-cxxfilt -n | FileCheck %s --check-prefixes=CHECK,CHECK-LE
// RUN: %clang -x c++ -fsyntax-only -target powerpc64le-unknown-freebsd13.0 -mcpu=pwr8 -ffreestanding -nostdlibinc -DNO_WARN_X86_INTRINSICS %s \
// RUN:   -fno-discard-value-names -mllvm -disable-llvm-optzns

// RUN: %clang -S -emit-llvm -target powerpc64-ibm-aix -mcpu=pwr8 -ffreestanding -DNO_WARN_X86_INTRINSICS %s \
// RUN:   -ffp-contract=off -fno-discard-value-names -mllvm -disable-llvm-optzns -o - | llvm-cxxfilt -n | FileCheck %s --check-prefixes=CHECK,CHECK-BE
// RUN: %clang -x c++ -fsyntax-only -target powerpc64-ibm-aix -mcpu=pwr8 -ffreestanding -DNO_WARN_X86_INTRINSICS %s \
// RUN:   -fno-discard-value-names -mllvm -disable-llvm-optzns
// RUN: %clang -S -emit-llvm -target powerpc64-ibm-aix -mcpu=pwr10 -ffreestanding -DNO_WARN_X86_INTRINSICS %s \
// RUN:   -ffp-contract=off -fno-discard-value-names -mllvm -disable-llvm-optzns -o - | llvm-cxxfilt -n | FileCheck %s --check-prefixes=CHECK,CHECK-P10-BE

#include <xmmintrin.h>

__m128 res, m1, m2;
__m64 res64, ms[2];
float fs[4];
int i, i2;
long long i64;

// CHECK-LE-DAG: @_mm_shuffle_pi16.__permute_selectors = internal constant [4 x i16] [i16 2312, i16 2826, i16 3340, i16 3854], align 2
// CHECK-BE-DAG: @_mm_shuffle_pi16.__permute_selectors = internal constant [4 x i16] [i16 1543, i16 1029, i16 515, i16 1], align 2

// CHECK-LE-DAG: @_mm_shuffle_ps.__permute_selectors = internal constant [4 x i32] [i32 50462976, i32 117835012, i32 185207048, i32 252579084], align 4
// CHECK-BE-DAG: @_mm_shuffle_ps.__permute_selectors = internal constant [4 x i32] [i32 66051, i32 67438087, i32 134810123, i32 202182159], align 4

void __attribute__((noinline))
test_add() {
  res = _mm_add_ps(m1, m2);
  res = _mm_add_ss(m1, m2);
}

// CHECK-LABEL: @test_add

// CHECK-LABEL: define available_externally <4 x float> @_mm_add_ps
// CHECK: fadd <4 x float>

// CHECK-LABEL: define available_externally <4 x float> @_mm_add_ss
// CHECK: call <4 x float> @vec_splat(float vector[4], unsigned int)(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, i32 noundef zeroext 0)
// CHECK: call <4 x float> @vec_splat(float vector[4], unsigned int)(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, i32 noundef zeroext 0)
// CHECK: fadd <4 x float>
// CHECK: call <4 x float> @vec_sel(float vector[4], float vector[4], unsigned int vector[4])(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, <4 x float> noundef %{{[0-9a-zA-Z_.]+}}, <4 x i32> noundef <i32 -1, i32 0, i32 0, i32 0>)

void __attribute__((noinline))
test_avg() {
  res64 = _mm_avg_pu16(ms[0], ms[1]);
  res64 = _mm_avg_pu8(ms[0], ms[1]);
}

// CHECK-LABEL: @test_avg

// CHECK-LABEL: define available_externally i64 @_mm_avg_pu16
// CHECK: call <2 x i64> @vec_splats(unsigned long long)
// CHECK: call <2 x i64> @vec_splats(unsigned long long)
// CHECK: call <8 x i16> @vec_avg(unsigned short vector[8], unsigned short vector[8])
// CHECK: extractelement <2 x i64> %{{[0-9a-zA-Z_.]+}}, i32 0

// CHECK-LABEL: define available_externally i64 @_mm_avg_pu8
// CHECK: call <2 x i64> @vec_splats(unsigned long long)
// CHECK: call <2 x i64> @vec_splats(unsigned long long)
// CHECK: call <16 x i8> @vec_avg(unsigned char vector[16], unsigned char vector[16])
// CHECK: extractelement <2 x i64> %{{[0-9a-zA-Z_.]+}}, i32 0

void __attribute__((noinline))
test_alt_name_avg() {
  res64 = _m_pavgw(ms[0], ms[1]);
  res64 = _m_pavgb(ms[0], ms[1]);
}

// CHECK-LABEL: @test_alt_name_avg

// CHECK-LABEL: define available_externally i64 @_m_pavgw
// CHECK: call i64 @_mm_avg_pu16

// CHECK-LABEL: define available_externally i64 @_m_pavgb
// CHECK: call i64 @_mm_avg_pu8

void __attribute__((noinline))
test_cmp() {
  res = _mm_cmpeq_ps(m1, m2);
  res = _mm_cmpeq_ss(m1, m2);
  res = _mm_cmpge_ps(m1, m2);
  res = _mm_cmpge_ss(m1, m2);
  res = _mm_cmpgt_ps(m1, m2);
  res = _mm_cmpgt_ss(m1, m2);
  res = _mm_cmple_ps(m1, m2);
  res = _mm_cmple_ss(m1, m2);
  res = _mm_cmplt_ps(m1, m2);
  res = _mm_cmplt_ss(m1, m2);
  res = _mm_cmpneq_ps(m1, m2);
  res = _mm_cmpneq_ss(m1, m2);
  res = _mm_cmpnge_ps(m1, m2);
  res = _mm_cmpnge_ss(m1, m2);
  res = _mm_cmpngt_ps(m1, m2);
  res = _mm_cmpngt_ss(m1, m2);
  res = _mm_cmpnle_ps(m1, m2);
  res = _mm_cmpnle_ss(m1, m2);
  res = _mm_cmpnlt_ps(m1, m2);
  res = _mm_cmpnlt_ss(m1, m2);
  res = _mm_cmpord_ps(m1, m2);
  res = _mm_cmpord_ss(m1, m2);
  res = _mm_cmpunord_ps(m1, m2);
  res = _mm_cmpunord_ss(m1, m2);
}

// CHECK-LABEL: @test_cmp

// CHECK-LABEL: define available_externally <4 x float> @_mm_cmpeq_ps
// CHECK: call <4 x i32> @vec_cmpeq(float vector[4], float vector[4])

// CHECK-LABEL: define available_externally <4 x float> @_mm_cmpeq_ss
// CHECK: call <4 x float> @vec_splat(float vector[4], unsigned int)(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, {{i32|i32 noundef zeroext}} 0)
// CHECK: call <4 x float> @vec_splat(float vector[4], unsigned int)(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, {{i32|i32 noundef zeroext}} 0)
// CHECK: call <4 x i32> @vec_cmpeq(float vector[4], float vector[4])
// CHECK: call <4 x float> @vec_sel(float vector[4], float vector[4], unsigned int vector[4])(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, <4 x float> noundef %{{[0-9a-zA-Z_.]+}}, <4 x i32> noundef <i32 -1, i32 0, i32 0, i32 0>)

// CHECK-LABEL: define available_externally <4 x float> @_mm_cmpge_ps
// CHECK: call <4 x i32> @vec_cmpge(float vector[4], float vector[4])

// CHECK-LABEL: define available_externally <4 x float> @_mm_cmpge_ss
// CHECK: call <4 x float> @vec_splat(float vector[4], unsigned int)(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, {{i32|i32 noundef zeroext}} 0)
// CHECK: call <4 x float> @vec_splat(float vector[4], unsigned int)(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, {{i32|i32 noundef zeroext}} 0)
// CHECK: call <4 x i32> @vec_cmpge(float vector[4], float vector[4])
// CHECK: call <4 x float> @vec_sel(float vector[4], float vector[4], unsigned int vector[4])(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, <4 x float> noundef %{{[0-9a-zA-Z_.]+}}, <4 x i32> noundef <i32 -1, i32 0, i32 0, i32 0>)

// CHECK-LABEL: define available_externally <4 x float> @_mm_cmpgt_ps
// CHECK: call <4 x i32> @vec_cmpgt(float vector[4], float vector[4])

// CHECK-LABEL: define available_externally <4 x float> @_mm_cmpgt_ss
// CHECK: call <4 x float> @vec_splat(float vector[4], unsigned int)(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, {{i32|i32 noundef zeroext}} 0)
// CHECK: call <4 x float> @vec_splat(float vector[4], unsigned int)(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, {{i32|i32 noundef zeroext}} 0)
// CHECK: call <4 x i32> @vec_cmpgt(float vector[4], float vector[4])
// CHECK: call <4 x float> @vec_sel(float vector[4], float vector[4], unsigned int vector[4])(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, <4 x float> noundef %{{[0-9a-zA-Z_.]+}}, <4 x i32> noundef <i32 -1, i32 0, i32 0, i32 0>)

// CHECK-LABEL: define available_externally <4 x float> @_mm_cmple_ps
// CHECK: call <4 x i32> @vec_cmple(float vector[4], float vector[4])

// CHECK-LABEL: define available_externally <4 x float> @_mm_cmple_ss
// CHECK: call <4 x float> @vec_splat(float vector[4], unsigned int)(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, {{i32|i32 noundef zeroext}} 0)
// CHECK: call <4 x float> @vec_splat(float vector[4], unsigned int)(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, {{i32|i32 noundef zeroext}} 0)
// CHECK: call <4 x i32> @vec_cmple(float vector[4], float vector[4])
// CHECK: call <4 x float> @vec_sel(float vector[4], float vector[4], unsigned int vector[4])(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, <4 x float> noundef %{{[0-9a-zA-Z_.]+}}, <4 x i32> noundef <i32 -1, i32 0, i32 0, i32 0>)

// CHECK-LABEL: define available_externally <4 x float> @_mm_cmplt_ps
// CHECK: call <4 x i32> @vec_cmplt(float vector[4], float vector[4])

// CHECK: @_mm_cmplt_ss
// CHECK: call <4 x float> @vec_splat(float vector[4], unsigned int)(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, i32 noundef zeroext 0)
// CHECK: call <4 x float> @vec_splat(float vector[4], unsigned int)(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, i32 noundef zeroext 0)
// CHECK: call <4 x i32> @vec_cmplt(float vector[4], float vector[4])
// CHECK: call <4 x float> @vec_sel(float vector[4], float vector[4], unsigned int vector[4])(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, <4 x float> noundef %{{[0-9a-zA-Z_.]+}}, <4 x i32> noundef <i32 -1, i32 0, i32 0, i32 0>)

// CHECK-LABEL: define available_externally <4 x float> @_mm_cmpneq_ps
// CHECK: call <4 x i32> @vec_cmpeq(float vector[4], float vector[4])
// CHECK: call <4 x float> @vec_nor(float vector[4], float vector[4])

// CHECK-LABEL: define available_externally <4 x float> @_mm_cmpneq_ss
// CHECK: call <4 x float> @vec_splat(float vector[4], unsigned int)(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, {{i32|i32 noundef zeroext}} 0)
// CHECK: call <4 x float> @vec_splat(float vector[4], unsigned int)(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, {{i32|i32 noundef zeroext}} 0)
// CHECK: call <4 x i32> @vec_cmpeq(float vector[4], float vector[4])
// CHECK: call <4 x float> @vec_nor(float vector[4], float vector[4])
// CHECK: call <4 x float> @vec_sel(float vector[4], float vector[4], unsigned int vector[4])(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, <4 x float> noundef %{{[0-9a-zA-Z_.]+}}, <4 x i32> noundef <i32 -1, i32 0, i32 0, i32 0>)

// CHECK-LABEL: define available_externally <4 x float> @_mm_cmpnge_ps
// CHECK: call <4 x i32> @vec_cmplt(float vector[4], float vector[4])

// CHECK-LABEL: define available_externally <4 x float> @_mm_cmpnge_ss
// CHECK: call <4 x float> @vec_splat(float vector[4], unsigned int)(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, {{i32|i32 noundef zeroext}} 0)
// CHECK: call <4 x float> @vec_splat(float vector[4], unsigned int)(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, {{i32|i32 noundef zeroext}} 0)
// CHECK: call <4 x i32> @vec_cmplt(float vector[4], float vector[4])
// CHECK: call <4 x float> @vec_sel(float vector[4], float vector[4], unsigned int vector[4])(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, <4 x float> noundef %{{[0-9a-zA-Z_.]+}}, <4 x i32> noundef <i32 -1, i32 0, i32 0, i32 0>)

// CHECK-LABEL: define available_externally <4 x float> @_mm_cmpngt_ps
// CHECK: call <4 x i32> @vec_cmple(float vector[4], float vector[4])

// CHECK-LABEL: define available_externally <4 x float> @_mm_cmpngt_ss
// CHECK: call <4 x float> @vec_splat(float vector[4], unsigned int)(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, {{i32|i32 noundef zeroext}} 0)
// CHECK: call <4 x float> @vec_splat(float vector[4], unsigned int)(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, {{i32|i32 noundef zeroext}} 0)
// CHECK: call <4 x i32> @vec_cmple(float vector[4], float vector[4])
// CHECK: call <4 x float> @vec_sel(float vector[4], float vector[4], unsigned int vector[4])(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, <4 x float> noundef %{{[0-9a-zA-Z_.]+}}, <4 x i32> noundef <i32 -1, i32 0, i32 0, i32 0>)

// CHECK-LABEL: define available_externally <4 x float> @_mm_cmpnle_ps
// CHECK: call <4 x i32> @vec_cmpgt(float vector[4], float vector[4])

// CHECK-LABEL: define available_externally <4 x float> @_mm_cmpnle_ss
// CHECK: call <4 x float> @vec_splat(float vector[4], unsigned int)(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, {{i32|i32 noundef zeroext}} 0)
// CHECK: call <4 x float> @vec_splat(float vector[4], unsigned int)(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, {{i32|i32 noundef zeroext}} 0)
// CHECK: call <4 x i32> @vec_cmpgt(float vector[4], float vector[4])
// CHECK: call <4 x float> @vec_sel(float vector[4], float vector[4], unsigned int vector[4])(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, <4 x float> noundef %{{[0-9a-zA-Z_.]+}}, <4 x i32> noundef <i32 -1, i32 0, i32 0, i32 0>)

// CHECK-LABEL: define available_externally <4 x float> @_mm_cmpnlt_ps
// CHECK: call <4 x i32> @vec_cmpge(float vector[4], float vector[4])

// CHECK-LABEL: define available_externally <4 x float> @_mm_cmpnlt_ss
// CHECK: call <4 x float> @vec_splat(float vector[4], unsigned int)(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, {{i32|i32 noundef zeroext}} 0)
// CHECK: call <4 x float> @vec_splat(float vector[4], unsigned int)(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, {{i32|i32 noundef zeroext}} 0)
// CHECK: call <4 x i32> @vec_cmpge(float vector[4], float vector[4])
// CHECK: call <4 x float> @vec_sel(float vector[4], float vector[4], unsigned int vector[4])(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, <4 x float> noundef %{{[0-9a-zA-Z_.]+}}, <4 x i32> noundef <i32 -1, i32 0, i32 0, i32 0>)

// CHECK-LABEL: define available_externally <4 x float> @_mm_cmpord_ps
// CHECK: call <4 x float> @vec_abs(float vector[4])(<4 x float> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: call <4 x float> @vec_abs(float vector[4])(<4 x float> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: call <4 x i32> @vec_cmpgt(unsigned int vector[4], unsigned int vector[4])(<4 x i32> noundef <i32 2139095040, i32 2139095040, i32 2139095040, i32 2139095040>, <4 x i32> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: call <4 x i32> @vec_cmpgt(unsigned int vector[4], unsigned int vector[4])(<4 x i32> noundef <i32 2139095040, i32 2139095040, i32 2139095040, i32 2139095040>, <4 x i32> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: call <4 x i32> @vec_and(unsigned int vector[4], unsigned int vector[4])

// CHECK-LABEL: define available_externally <4 x float> @_mm_cmpord_ss
// CHECK: call <4 x float> @vec_abs(float vector[4])
// CHECK: call <4 x float> @vec_abs(float vector[4])
// CHECK: call <4 x i32> @vec_cmpgt(unsigned int vector[4], unsigned int vector[4])(<4 x i32> noundef <i32 2139095040, i32 2139095040, i32 2139095040, i32 2139095040>, <4 x i32> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: call <4 x i32> @vec_cmpgt(unsigned int vector[4], unsigned int vector[4])(<4 x i32> noundef <i32 2139095040, i32 2139095040, i32 2139095040, i32 2139095040>, <4 x i32> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: call <4 x i32> @vec_and(unsigned int vector[4], unsigned int vector[4])(<4 x i32> noundef %{{[0-9a-zA-Z_.]+}}, <4 x i32> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: call <4 x float> @vec_sel(float vector[4], float vector[4], unsigned int vector[4])(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, <4 x float> noundef %{{[0-9a-zA-Z_.]+}}, <4 x i32> noundef <i32 -1, i32 0, i32 0, i32 0>)

// CHECK-LABEL: define available_externally <4 x float> @_mm_cmpunord_ps
// CHECK: call <4 x float> @vec_abs(float vector[4])
// CHECK: call <4 x float> @vec_abs(float vector[4])
// CHECK: call <4 x i32> @vec_cmpgt(unsigned int vector[4], unsigned int vector[4])(<4 x i32> noundef %{{[0-9a-zA-Z_.]+}}, <4 x i32> noundef <i32 2139095040, i32 2139095040, i32 2139095040, i32 2139095040>)
// CHECK: call <4 x i32> @vec_cmpgt(unsigned int vector[4], unsigned int vector[4])(<4 x i32> noundef %{{[0-9a-zA-Z_.]+}}, <4 x i32> noundef <i32 2139095040, i32 2139095040, i32 2139095040, i32 2139095040>)
// CHECK: call <4 x i32> @vec_or(unsigned int vector[4], unsigned int vector[4])

// CHECK-LABEL: define available_externally <4 x float> @_mm_cmpunord_ss
// CHECK: call <4 x float> @vec_abs(float vector[4])
// CHECK: call <4 x float> @vec_abs(float vector[4])
// CHECK: call <4 x i32> @vec_cmpgt(unsigned int vector[4], unsigned int vector[4])(<4 x i32> noundef %{{[0-9a-zA-Z_.]+}}, <4 x i32> noundef <i32 2139095040, i32 2139095040, i32 2139095040, i32 2139095040>)
// CHECK: call <4 x i32> @vec_cmpgt(unsigned int vector[4], unsigned int vector[4])(<4 x i32> noundef %{{[0-9a-zA-Z_.]+}}, <4 x i32> noundef <i32 2139095040, i32 2139095040, i32 2139095040, i32 2139095040>)
// CHECK: call <4 x i32> @vec_or(unsigned int vector[4], unsigned int vector[4])
// CHECK: call <4 x float> @vec_sel(float vector[4], float vector[4], unsigned int vector[4])(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, <4 x float> noundef %{{[0-9a-zA-Z_.]+}}, <4 x i32> noundef <i32 -1, i32 0, i32 0, i32 0>)

void __attribute__((noinline))
test_comi() {
  i = _mm_comieq_ss(m1, m2);
  i = _mm_comige_ss(m1, m2);
  i = _mm_comigt_ss(m1, m2);
  i = _mm_comile_ss(m1, m2);
  i = _mm_comilt_ss(m1, m2);
  i = _mm_comineq_ss(m1, m2);
}

// CHECK-LABEL: @test_comi

// CHECK-LABEL: define available_externally signext i32 @_mm_comieq_ss
// CHECK: %[[VAL1:[0-9a-zA-Z_.]+]] = extractelement <4 x float> %{{[0-9a-zA-Z_.]+}}, i32 0
// CHECK: %[[VAL2:[0-9a-zA-Z_.]+]] = extractelement <4 x float> %{{[0-9a-zA-Z_.]+}}, i32 0
// CHECK: %[[CMP:[0-9a-zA-Z_.]+]] = fcmp oeq float %[[VAL1]], %[[VAL2]]
// CHECK: zext i1 %[[CMP]] to i32

// CHECK-LABEL: define available_externally signext i32 @_mm_comige_ss
// CHECK: %[[VAL1:[0-9a-zA-Z_.]+]] = extractelement <4 x float> %{{[0-9a-zA-Z_.]+}}, i32 0
// CHECK: %[[VAL2:[0-9a-zA-Z_.]+]] = extractelement <4 x float> %{{[0-9a-zA-Z_.]+}}, i32 0
// CHECK: %[[CMP:[0-9a-zA-Z_.]+]] = fcmp oge float %[[VAL1]], %[[VAL2]]
// CHECK: zext i1 %[[CMP]] to i32

// CHECK-LABEL: define available_externally signext i32 @_mm_comigt_ss
// CHECK: %[[VAL1:[0-9a-zA-Z_.]+]] = extractelement <4 x float> %{{[0-9a-zA-Z_.]+}}, i32 0
// CHECK: %[[VAL2:[0-9a-zA-Z_.]+]] = extractelement <4 x float> %{{[0-9a-zA-Z_.]+}}, i32 0
// CHECK: %[[CMP:[0-9a-zA-Z_.]+]] = fcmp ogt float %[[VAL1]], %[[VAL2]]
// CHECK: zext i1 %[[CMP]] to i32

// CHECK-LABEL: define available_externally signext i32 @_mm_comile_ss
// CHECK: %[[VAL1:[0-9a-zA-Z_.]+]] = extractelement <4 x float> %{{[0-9a-zA-Z_.]+}}, i32 0
// CHECK: %[[VAL2:[0-9a-zA-Z_.]+]] = extractelement <4 x float> %{{[0-9a-zA-Z_.]+}}, i32 0
// CHECK: %[[CMP:[0-9a-zA-Z_.]+]] = fcmp ole float %[[VAL1]], %[[VAL2]]
// CHECK: zext i1 %[[CMP]] to i32

// CHECK-LABEL: define available_externally signext i32 @_mm_comilt_ss
// CHECK: %[[VAL1:[0-9a-zA-Z_.]+]] = extractelement <4 x float> %{{[0-9a-zA-Z_.]+}}, i32 0
// CHECK: %[[VAL2:[0-9a-zA-Z_.]+]] = extractelement <4 x float> %{{[0-9a-zA-Z_.]+}}, i32 0
// CHECK: %[[CMP:[0-9a-zA-Z_.]+]] = fcmp olt float %[[VAL1]], %[[VAL2]]
// CHECK: zext i1 %[[CMP]] to i32

// CHECK-LABEL: define available_externally signext i32 @_mm_comineq_ss
// CHECK: %[[VAL1:[0-9a-zA-Z_.]+]] = extractelement <4 x float> %{{[0-9a-zA-Z_.]+}}, i32 0
// CHECK: %[[VAL2:[0-9a-zA-Z_.]+]] = extractelement <4 x float> %{{[0-9a-zA-Z_.]+}}, i32 0
// CHECK: %[[CMP:[0-9a-zA-Z_.]+]] = fcmp une float %[[VAL1]], %[[VAL2]]
// CHECK: zext i1 %[[CMP]] to i32

void __attribute__((noinline))
test_convert() {
  res = _mm_cvt_pi2ps(m1, ms[1]);
  res64 = _mm_cvt_ps2pi(m1);
  res = _mm_cvt_si2ss(m1, i);
  i = _mm_cvt_ss2si(m1);
  res = _mm_cvtpi16_ps(ms[0]);
  res = _mm_cvtpi32_ps(m1, ms[1]);
  res = _mm_cvtpi32x2_ps(ms[0], ms[1]);
  res = _mm_cvtpi8_ps(ms[0]);
  res64 = _mm_cvtps_pi16(m1);
  res64 = _mm_cvtps_pi32(m1);
  res64 = _mm_cvtps_pi8(m1);
  res = _mm_cvtpu16_ps(ms[0]);
  res = _mm_cvtpu8_ps(ms[0]);
  res = _mm_cvtsi32_ss(m1, i);
  res = _mm_cvtsi64_ss(m1, i64);
  fs[0] = _mm_cvtss_f32(m1);
  i = _mm_cvtss_si32(m1);
  i64 = _mm_cvtss_si64(m1);
  res64 = _mm_cvtt_ps2pi(m1);
  i = _mm_cvtt_ss2si(m1);
  res64 = _mm_cvttps_pi32(m1);
  i = _mm_cvttss_si32(m1);
  i64 = _mm_cvttss_si64(m1);
}

// CHECK-LABEL: @test_convert

// CHECK-LABEL: define available_externally <4 x float> @_mm_cvt_pi2ps
// CHECK: call <4 x float> @_mm_cvtpi32_ps

// CHECK-LABEL: define available_externally i64 @_mm_cvt_ps2pi
// CHECK: call i64 @_mm_cvtps_pi32

// CHECK-LABEL: define available_externally <4 x float> @_mm_cvt_si2ss
// CHECK: call <4 x float> @_mm_cvtsi32_ss

// CHECK-LABEL: define available_externally signext i32 @_mm_cvt_ss2si
// CHECK: call signext i32 @_mm_cvtss_si32

// CHECK-LABEL: define available_externally <4 x float> @_mm_cvtpi16_ps
// CHECK: call <4 x i32> @vec_vupklsh(short vector[8])
// CHECK: call <4 x float> @llvm.ppc.altivec.vcfsx(<4 x i32> %{{[0-9a-zA-Z_.]+}}, i32 0)

// CHECK-LABEL: define available_externally <4 x float> @_mm_cvtpi32_ps
// CHECK: call <4 x float> @llvm.ppc.altivec.vcfsx(<4 x i32> %{{[0-9a-zA-Z_.]+}}, i32 0)

// CHECK-LABEL: define available_externally <4 x float> @_mm_cvtpi32x2_ps
// CHECK: call <4 x float> @llvm.ppc.altivec.vcfsx(<4 x i32> %{{[0-9a-zA-Z_.]+}}, i32 0)

// CHECK-LABEL: define available_externally <4 x float> @_mm_cvtpi8_ps
// CHECK: call <8 x i16> @vec_vupkhsb(signed char vector[16])
// CHECK: call <4 x i32> @vec_vupkhsh(short vector[8])
// CHECK: call <4 x float> @llvm.ppc.altivec.vcfsx(<4 x i32> %{{[0-9a-zA-Z_.]+}}, i32 0)

// CHECK-LABEL: define available_externally i64 @_mm_cvtps_pi16
// CHECK: call <4 x float> @vec_rint(float vector[4])
// CHECK: call <4 x i32> @llvm.ppc.altivec.vctsxs(<4 x float> %{{[0-9a-zA-Z_.]+}}, i32 0)
// CHECK: call <8 x i16> @vec_pack(int vector[4], int vector[4])
// CHECK: extractelement <2 x i64> %{{[0-9a-zA-Z_.]+}}, i32 0

// CHECK-LABEL: define available_externally i64 @_mm_cvtps_pi32
// CHECK: call <2 x i64> @vec_splat(long long vector[2], unsigned int)(<2 x i64> noundef %{{[0-9a-zA-Z_.]+}}, i32 noundef zeroext 0)
// CHECK: call <4 x float> @vec_rint(float vector[4])
// CHECK: call <4 x i32> @llvm.ppc.altivec.vctsxs(<4 x float> %{{[0-9a-zA-Z_.]+}}, i32 0)
// CHECK: extractelement <2 x i64> %{{[0-9a-zA-Z_.]+}}, i32 0

// CHECK-LABEL: define available_externally i64 @_mm_cvtps_pi8
// CHECK: call <4 x float> @vec_rint(float vector[4])
// CHECK: call <4 x i32> @llvm.ppc.altivec.vctsxs(<4 x float> %{{[0-9a-zA-Z_.]+}}, i32 0)
// CHECK: call <8 x i16> @vec_pack(int vector[4], int vector[4])(<4 x i32> noundef %{{[0-9a-zA-Z_.]+}}, <4 x i32> noundef zeroinitializer)
// CHECK: call <16 x i8> @vec_pack(short vector[8], short vector[8])
// CHECK: extractelement <2 x i64> %{{[0-9a-zA-Z_.]+}}, i32 0

// CHECK-LABEL: define available_externally <4 x float> @_mm_cvtpu16_ps
// CHECK-LE: call <8 x i16> @vec_mergel(unsigned short vector[8], unsigned short vector[8])(<8 x i16> noundef %{{[0-9a-zA-Z_.]+}}, <8 x i16> noundef zeroinitializer)
// CHECK-BE: call <8 x i16> @vec_mergel(unsigned short vector[8], unsigned short vector[8])(<8 x i16> noundef zeroinitializer, <8 x i16> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: call <4 x float> @llvm.ppc.altivec.vcfux(<4 x i32> %{{[0-9a-zA-Z_.]+}}, i32 0)

// CHECK-LABEL: define available_externally <4 x float> @_mm_cvtpu8_ps
// CHECK-BE: call <16 x i8> @vec_mergel(unsigned char vector[16], unsigned char vector[16])(<16 x i8> noundef zeroinitializer, <16 x i8> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK-BE: call <8 x i16> @vec_mergeh(unsigned short vector[8], unsigned short vector[8])(<8 x i16> noundef zeroinitializer, <8 x i16> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK-LE: call <16 x i8> @vec_mergel(unsigned char vector[16], unsigned char vector[16])(<16 x i8> noundef %{{[0-9a-zA-Z_.]+}}, <16 x i8> noundef zeroinitializer)
// CHECK-LE: call <8 x i16> @vec_mergeh(unsigned short vector[8], unsigned short vector[8])(<8 x i16> noundef %{{[0-9a-zA-Z_.]+}}, <8 x i16> noundef zeroinitializer)
// CHECK: call <4 x float> @llvm.ppc.altivec.vcfux(<4 x i32> %{{[0-9a-zA-Z_.]+}}, i32 0)

// CHECK-LABEL: define available_externally <4 x float> @_mm_cvtsi32_ss
// CHECK: sitofp i32 %{{[0-9a-zA-Z_.]+}} to float
// CHECK: insertelement <4 x float> %{{[0-9a-zA-Z_.]+}}, float %{{[0-9a-zA-Z_.]+}}, i32 0

// CHECK-LABEL: define available_externally <4 x float> @_mm_cvtsi64_ss
// CHECK: sitofp i64 %{{[0-9a-zA-Z_.]+}} to float
// CHECK: insertelement <4 x float> %{{[0-9a-zA-Z_.]+}}, float %{{[0-9a-zA-Z_.]+}}, i32 0

// CHECK-LABEL: define available_externally float @_mm_cvtss_f32
// CHECK: extractelement <4 x float> %{{[0-9a-zA-Z_.]+}}, i32 0

// CHECK-LABEL: define available_externally signext i32 @_mm_cvtss_si32
// CHECK-LE: %[[VEC:[0-9a-zA-Z_.]+]] = call { <4 x float>, i32, double } asm "xxsldwi ${0:x},${0:x},${0:x},3;\0Axscvspdp ${2:x},${0:x};\0Afctiw  $2,$2;\0Amfvsrd  $1,${2:x};\0A", "=^wa,=r,=f,0"
// CHECK-BE: %[[VEC:[0-9a-zA-Z_.]+]] = call { <4 x float>, i32, double } asm "xscvspdp ${2:x},${0:x};\0Afctiw  $2,$2;\0Amfvsrd  $1,${2:x};\0A", "=^wa,=r,=f,0"
// CHECK-P10-LE: %[[VEC:[0-9a-zA-Z_.]+]] = call { <4 x float>, i32, double } asm "xxsldwi ${0:x},${0:x},${0:x},3;\0Axscvspdp ${2:x},${0:x};\0Afctiw  $2,$2;\0Amfvsrd  $1,${2:x};\0A", "=^wa,=r,=f,0"
// CHECK-P10-BE: %[[VEC:[0-9a-zA-Z_.]+]] = call { <4 x float>, i32, double } asm "xscvspdp ${2:x},${0:x};\0Afctiw  $2,$2;\0Amfvsrd  $1,${2:x};\0A", "=^wa,=r,=f,0"
// CHECK: extractvalue { <4 x float>, i32, double } %[[VEC]], 0
// CHECK: extractvalue { <4 x float>, i32, double } %[[VEC]], 1
// CHECK: extractvalue { <4 x float>, i32, double } %[[VEC]], 2

// CHECK-LABEL: define available_externally i64 @_mm_cvtss_si64
// CHECK-LE: %[[VEC:[0-9a-zA-Z_.]+]] = call { <4 x float>, i64, double } asm "xxsldwi ${0:x},${0:x},${0:x},3;\0Axscvspdp ${2:x},${0:x};\0Afctid  $2,$2;\0Amfvsrd  $1,${2:x};\0A", "=^wa,=r,=f,0"
// CHECK-BE: %[[VEC:[0-9a-zA-Z_.]+]] = call { <4 x float>, i64, double } asm "xscvspdp ${2:x},${0:x};\0Afctid  $2,$2;\0Amfvsrd  $1,${2:x};\0A", "=^wa,=r,=f,0"
// CHECK: extractvalue { <4 x float>, i64, double } %[[VEC]], 0
// CHECK: extractvalue { <4 x float>, i64, double } %[[VEC]], 1
// CHECK: extractvalue { <4 x float>, i64, double } %[[VEC]], 2

// CHECK-LABEL: define available_externally i64 @_mm_cvtt_ps2pi
// CHECK: call i64 @_mm_cvttps_pi32(<4 x float> noundef %{{[0-9a-zA-Z_.]+}})

// CHECK-LABEL: define available_externally signext i32 @_mm_cvtt_ss2si
// CHECK: call signext i32 @_mm_cvttss_si32(<4 x float> noundef %{{[0-9a-zA-Z_.]+}})

// CHECK-LABEL: define available_externally i64 @_mm_cvttps_pi32
// CHECK: call <2 x i64> @vec_splat(long long vector[2], unsigned int)(<2 x i64> noundef %{{[0-9a-zA-Z_.]+}}, i32 noundef zeroext 0)
// CHECK: call <4 x i32> @llvm.ppc.altivec.vctsxs(<4 x float> %{{[0-9a-zA-Z_.]+}}, i32 0)
// CHECK: extractelement <2 x i64> %{{[0-9a-zA-Z_.]+}}, i32 0

// CHECK-LABEL: define available_externally signext i32 @_mm_cvttss_si32
// CHECK: extractelement <4 x float> %{{[0-9a-zA-Z_.]+}}, i32 0
// CHECK: fptosi float %{{[0-9a-zA-Z_.]+}} to i32

// CHECK-LABEL: define available_externally i64 @_mm_cvttss_si64
// CHECK: extractelement <4 x float> %{{[0-9a-zA-Z_.]+}}, i32 0
// CHECK: fptosi float %{{[0-9a-zA-Z_.]+}} to i64

void __attribute__((noinline))
test_div() {
  res = _mm_div_ps(m1, m2);
  res = _mm_div_ss(m1, m2);
}

// CHECK-LABEL: @test_div

// CHECK-LABEL: define available_externally <4 x float> @_mm_div_ps
// CHECK: fdiv <4 x float>

// CHECK-LABEL: define available_externally <4 x float> @_mm_div_ss
// CHECK: call <4 x float> @vec_splat(float vector[4], unsigned int)(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, i32 noundef zeroext 0)
// CHECK: call <4 x float> @vec_splat(float vector[4], unsigned int)(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, i32 noundef zeroext 0)
// CHECK: fdiv <4 x float>
// CHECK: call <4 x float> @vec_sel(float vector[4], float vector[4], unsigned int vector[4])(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, <4 x float> noundef %{{[0-9a-zA-Z_.]+}}, <4 x i32> noundef <i32 -1, i32 0, i32 0, i32 0>)

void __attribute__((noinline))
test_extract() {
  i = _mm_extract_pi16(ms[0], i2);
  i = _m_pextrw(ms[0], i2);
}

// CHECK-LABEL: @test_extract

// CHECK-LABEL: define available_externally signext i32 @_mm_extract_pi16
// CHECK: and i32 %{{[0-9a-zA-Z_.]+}}, 3
// CHECK-BE: sub i32 3, %{{[0-9a-zA-Z_.]+}}
// CHECK: %[[MUL:[0-9a-zA-Z_.]+]] = mul i32 %{{[0-9a-zA-Z_.]+}}, 16
// CHECK: %[[EXT:[0-9a-zA-Z_.]+]] = zext i32 %[[MUL]] to i64
// CHECK: %[[SHR:[0-9a-zA-Z_.]+]] = lshr i64 %{{[0-9a-zA-Z_.]+}}, %[[EXT]]
// CHECK: %[[AND:[0-9a-zA-Z_.]+]] = and i64 %[[SHR]], 65535
// CHECK: trunc i64 %[[AND]] to i32

// CHECK-LABEL: define available_externally signext i32 @_m_pextrw
// CHECK: call signext i32 @_mm_extract_pi16

void __attribute__((noinline))
test_insert() {
  res64 = _mm_insert_pi16(ms[0], i, i2);
  res64 = _m_pinsrw(ms[0], i, i2);
}

// CHECK-LABEL: @test_insert

// CHECK-LABEL: define available_externally i64 @_mm_insert_pi16
// CHECK: %[[AND:[0-9a-zA-Z_.]+]] = and i32 %{{[0-9a-zA-Z_.]+}}, 3
// CHECK: mul nsw i32 %[[AND]], 16
// CHECK: %[[EXT:[0-9a-zA-Z_.]+]] = sext i32 %{{[0-9a-zA-Z_.]+}} to i64
// CHECK: %[[EXT2:[0-9a-zA-Z_.]+]] = zext i32 %{{[0-9a-zA-Z_.]+}} to i64
// CHECK: shl i64 %[[EXT]], %[[EXT2]]
// CHECK: %[[EXT3:[0-9a-zA-Z_.]+]] = zext i32 %{{[0-9a-zA-Z_.]+}} to i64
// CHECK: shl i64 65535, %[[EXT3]]
// CHECK: %[[XOR:[0-9a-zA-Z_.]+]] = xor i64 %{{[0-9a-zA-Z_.]+}}, -1
// CHECK: %[[AND2:[0-9a-zA-Z_.]+]] = and i64 %{{[0-9a-zA-Z_.]+}}, %[[XOR]]
// CHECK: %[[AND3:[0-9a-zA-Z_.]+]] = and i64 %{{[0-9a-zA-Z_.]+}}, %{{[0-9a-zA-Z_.]+}}
// CHECK: or i64 %[[AND2]], %[[AND3]]

// CHECK-LABEL: define available_externally i64 @_m_pinsrw
// CHECK: call i64 @_mm_insert_pi16

void __attribute__((noinline))
test_load() {
  res = _mm_load_ps(fs);
  res = _mm_load_ps1(fs);
  res = _mm_load_ss(fs);
  res = _mm_load1_ps(fs);
  res = _mm_loadh_pi(m1, &ms[0]);
  res = _mm_loadl_pi(m1, &ms[0]);
  res = _mm_loadr_ps(fs);
  res = _mm_loadu_ps(fs);
}

// CHECK-LABEL: @test_load

// CHECK-LABEL: define available_externally <4 x float> @_mm_load_ps
// CHECK: call <4 x float> @vec_ld(long, float vector[4] const*)

// CHECK-LABEL: define available_externally <4 x float> @_mm_load_ps1
// CHECK: call <4 x float> @_mm_load1_ps

// CHECK-LABEL: define available_externally <4 x float> @_mm_load_ss
// CHECK: call <4 x float> @_mm_set_ss

// CHECK-LABEL: define available_externally <4 x float> @_mm_load1_ps
// CHECK: call <4 x float> @_mm_set1_ps

// CHECK-LABEL: define available_externally <4 x float> @_mm_loadh_pi
// CHECK: call <2 x i64> @vec_splats(unsigned long long)
// CHECK: %[[VAL:[0-9a-zA-Z_.]+]] = extractelement <2 x i64> %{{[0-9a-zA-Z_.]+}}, i32 1
// CHECK: insertelement <2 x i64> %{{[0-9a-zA-Z_.]+}}, i64 %[[VAL]], i32 1

// CHECK-LABEL: define available_externally <4 x float> @_mm_loadl_pi
// CHECK: call <2 x i64> @vec_splats(unsigned long long)
// CHECK: %[[EXT:[0-9a-zA-Z_.]+]] = extractelement <2 x i64> %{{[0-9a-zA-Z_.]+}}, i32 0
// CHECK: insertelement <2 x i64> %{{[0-9a-zA-Z_.]+}}, i64 %[[EXT]], i32 0

// CHECK-LABEL: define available_externally <4 x float> @_mm_loadr_ps
// CHECK: call <4 x float> @vec_ld(long, float vector[4] const*)
// CHECK: call <4 x float> @vec_perm(float vector[4], float vector[4], unsigned char vector[16])(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, <4 x float> noundef %{{[0-9a-zA-Z_.]+}}, <16 x i8> noundef <i8 28, i8 29, i8 30, i8 31, i8 24, i8 25, i8 26, i8 27, i8 20, i8 21, i8 22, i8 23, i8 16, i8 17, i8 18, i8 19>)

// CHECK-LABEL: define available_externally <4 x float> @_mm_loadu_ps
// CHECK: call <4 x float> @vec_vsx_ld(int, float const*)

void __attribute__((noinline))
test_logic() {
  res = _mm_or_ps(m1, m2);
  res = _mm_and_ps(m1, m2);
  res = _mm_andnot_ps(m1, m2);
  res = _mm_xor_ps(m1, m2);
}

// CHECK-LABEL: @test_logic

// CHECK-LABEL: define available_externally <4 x float> @_mm_or_ps
// CHECK: call <4 x float> @vec_or(float vector[4], float vector[4])

// CHECK-LABEL: define available_externally <4 x float> @_mm_and_ps
// CHECK: call <4 x float> @vec_and(float vector[4], float vector[4])

// CHECK-LABEL: define available_externally <4 x float> @_mm_andnot_ps
// CHECK: call <4 x float> @vec_andc(float vector[4], float vector[4])

// CHECK-LABEL: define available_externally <4 x float> @_mm_xor_ps
// CHECK: call <4 x float> @vec_xor(float vector[4], float vector[4])

void __attribute__((noinline))
test_max() {
  res = _mm_max_ps(m1, m2);
  res = _mm_max_ss(m1, m2);
  res64 = _mm_max_pi16(ms[0], ms[1]);
  res64 = _mm_max_pu8(ms[0], ms[1]);
}

// CHECK-LABEL: @test_max

// CHECK-LABEL: define available_externally <4 x float> @_mm_max_ps
// CHECK: call <4 x i32> @vec_cmpgt(float vector[4], float vector[4])
// CHECK: call <4 x float> @vec_sel(float vector[4], float vector[4], bool vector[4])

// CHECK-LABEL: define available_externally <4 x float> @_mm_max_ss
// CHECK: call <4 x float> @vec_splat(float vector[4], unsigned int)(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, {{i32|i32 noundef zeroext}} 0)
// CHECK: call <4 x float> @vec_splat(float vector[4], unsigned int)(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, {{i32|i32 noundef zeroext}} 0)
// CHECK: call <4 x float> @vec_max(float vector[4], float vector[4])
// CHECK: call <4 x float> @vec_sel(float vector[4], float vector[4], unsigned int vector[4])(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, <4 x float> noundef %{{[0-9a-zA-Z_.]+}}, <4 x i32> noundef <i32 -1, i32 0, i32 0, i32 0>)

// CHECK-LABEL: define available_externally i64 @_mm_max_pi16
// CHECK: call <2 x i64> @vec_splats(unsigned long long)
// CHECK: call <2 x i64> @vec_splats(unsigned long long)
// CHECK: call <8 x i16> @vec_cmpgt(short vector[8], short vector[8])
// CHECK: call <8 x i16> @vec_sel(short vector[8], short vector[8], bool vector[8])
// CHECK: %[[CAST:[0-9a-zA-Z_.]+]] = bitcast <8 x i16> %{{[0-9a-zA-Z_.]+}} to <2 x i64>
// CHECK: extractelement <2 x i64> %[[CAST]], i32 0

// CHECK-LABEL: define available_externally i64 @_mm_max_pu8
// CHECK: call <2 x i64> @vec_splats(unsigned long long)
// CHECK: call <2 x i64> @vec_splats(unsigned long long)
// CHECK: call <16 x i8> @vec_cmpgt(unsigned char vector[16], unsigned char vector[16])
// CHECK: call <16 x i8> @vec_sel(unsigned char vector[16], unsigned char vector[16], bool vector[16])
// CHECK: %[[CAST:[0-9a-zA-Z_.]+]] = bitcast <16 x i8> %{{[0-9a-zA-Z_.]+}} to <2 x i64>
// CHECK: extractelement <2 x i64> %[[CAST]], i32 0

void __attribute__((noinline))
test_alt_name_max() {
  res64 = _m_pmaxsw(ms[0], ms[1]);
  res64 = _m_pmaxub(ms[0], ms[1]);
}

// CHECK-LABEL: @test_alt_name_max

// CHECK-LABEL: define available_externally i64 @_m_pmaxsw
// CHECK: call i64 @_mm_max_pi16

// CHECK-LABEL: define available_externally i64 @_m_pmaxub
// CHECK: call i64 @_mm_max_pu8

void __attribute__((noinline))
test_min() {
  res = _mm_min_ps(m1, m2);
  res = _mm_min_ss(m1, m2);
  res64 = _mm_min_pi16(ms[0], ms[1]);
  res64 = _mm_min_pu8(ms[0], ms[1]);
}

// CHECK-LABEL: @test_min

// CHECK-LABEL: define available_externally <4 x float> @_mm_min_ps
// CHECK: call <4 x i32> @vec_cmpgt(float vector[4], float vector[4])
// CHECK: call <4 x float> @vec_sel(float vector[4], float vector[4], bool vector[4])

// CHECK-LABEL: define available_externally <4 x float> @_mm_min_ss
// CHECK: call <4 x float> @vec_splat(float vector[4], unsigned int)(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, {{i32|i32 noundef zeroext}} 0)
// CHECK: call <4 x float> @vec_splat(float vector[4], unsigned int)(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, {{i32|i32 noundef zeroext}} 0)
// CHECK: call <4 x float> @vec_min(float vector[4], float vector[4])
// CHECK: call <4 x float> @vec_sel(float vector[4], float vector[4], unsigned int vector[4])(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, <4 x float> noundef %{{[0-9a-zA-Z_.]+}}, <4 x i32> noundef <i32 -1, i32 0, i32 0, i32 0>)

// CHECK-LABEL: define available_externally i64 @_mm_min_pi16
// CHECK: call <2 x i64> @vec_splats(unsigned long long)
// CHECK: call <2 x i64> @vec_splats(unsigned long long)
// CHECK: call <8 x i16> @vec_cmplt(short vector[8], short vector[8])
// CHECK: call <8 x i16> @vec_sel(short vector[8], short vector[8], bool vector[8])
// CHECK: %[[CAST:[0-9a-zA-Z_.]+]] = bitcast <8 x i16> %{{[0-9a-zA-Z_.]+}} to <2 x i64>
// CHECK: extractelement <2 x i64> %[[CAST]], i32 0

// CHECK-LABEL: define available_externally i64 @_mm_min_pu8
// CHECK: call <2 x i64> @vec_splats(unsigned long long)
// CHECK: call <2 x i64> @vec_splats(unsigned long long)
// CHECK: call <16 x i8> @vec_cmplt(unsigned char vector[16], unsigned char vector[16])
// CHECK: call <16 x i8> @vec_sel(unsigned char vector[16], unsigned char vector[16], bool vector[16])
// CHECK: %[[CAST:[0-9a-zA-Z_.]+]] = bitcast <16 x i8> %{{[0-9a-zA-Z_.]+}} to <2 x i64>
// CHECK: extractelement <2 x i64> %[[CAST]], i32 0

void __attribute__((noinline))
test_alt_name_min() {
  res64 = _m_pminsw(ms[0], ms[1]);
    res64 = _m_pminub(ms[0], ms[1]);
}

// CHECK-LABEL: @test_alt_name_min

// CHECK-LABEL: define available_externally i64 @_m_pminsw
// CHECK: call i64 @_mm_min_pi16

// CHECK-LABEL: define available_externally i64 @_m_pminub
// CHECK: call i64 @_mm_min_pu8

void __attribute__((noinline))
test_move() {
  _mm_maskmove_si64(ms[0], ms[1], (char *)&res64);
  res = _mm_move_ss(m1, m2);
  res = _mm_movehl_ps(m1, m2);
  res = _mm_movelh_ps(m1, m2);
  i = _mm_movemask_pi8(ms[0]);
  i = _mm_movemask_ps(m1);
}

// CHECK-LABEL: @test_move

// CHECK-LABEL: define available_externally void @_mm_maskmove_si64
// CHECK: store i64 -9187201950435737472, ptr %{{[0-9a-zA-Z_.]+}}, align 8
// CHECK: %[[AND:[0-9a-zA-Z_.]+]] = and i64
// CHECK: call i64 @_mm_cmpeq_pi8(i64 noundef %[[AND]], i64 noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: %[[XOR:[0-9a-zA-Z_.]+]] = xor i64 %{{[0-9a-zA-Z_.]+}}, -1
// CHECK: %[[AND2:[0-9a-zA-Z_.]+]] = and i64 %{{[0-9a-zA-Z_.]+}}, %[[XOR]]
// CHECK: %[[AND3:[0-9a-zA-Z_.]+]] = and i64
// CHECK: or i64 %[[AND2]], %[[AND3]]

// CHECK-LABEL: define available_externally <4 x float> @_mm_move_ss
// CHECK: call <4 x float> @vec_sel(float vector[4], float vector[4], unsigned int vector[4])(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, <4 x float> noundef %{{[0-9a-zA-Z_.]+}}, <4 x i32> noundef <i32 -1, i32 0, i32 0, i32 0>)

// CHECK-LABEL: define available_externally <4 x float> @_mm_movehl_ps
// CHECK: call <2 x i64> @vec_mergel(unsigned long long vector[2], unsigned long long vector[2])

// CHECK-LABEL: define available_externally <4 x float> @_mm_movelh_ps
// CHECK: call <2 x i64> @vec_mergeh(unsigned long long vector[2], unsigned long long vector[2])

// CHECK-LABEL: define available_externally signext i32 @_mm_movemask_pi8
// CHECK-LE: store i64 2269495618449464, ptr %{{[0-9a-zA-Z_.]+}}, align 8
// CHECK-BE: store i64 4048780183313844224, ptr %{{[0-9a-zA-Z_.]+}}, align 8
// CHECK: %[[CALL:[0-9a-zA-Z_.]+]] = call i64 @llvm.ppc.bpermd
// CHECK: trunc i64 %[[CALL]] to i32

// CHECK-LABEL: define available_externally signext i32 @_mm_movemask_ps
// CHECK-LE: call <2 x i64> @vec_vbpermq(unsigned char vector[16], unsigned char vector[16])(<16 x i8> noundef %{{[0-9a-zA-Z_.]+}}, <16 x i8> noundef bitcast (<4 x i32> <i32 2113632, i32 -2139062144, i32 -2139062144, i32 -2139062144> to <16 x i8>))
// CHECK-LE: extractelement <2 x i64> %{{[0-9a-zA-Z_.]+}}, i32 1
// CHECK-LE: trunc i64 %[[EXT]] to i32
// CHECK-BE: call <2 x i64> @vec_vbpermq(unsigned char vector[16], unsigned char vector[16])(<16 x i8> noundef %{{[0-9a-zA-Z_.]+}}, <16 x i8> noundef bitcast (<4 x i32> <i32 -2139062144, i32 -2139062144, i32 -2139062144, i32 2113632> to <16 x i8>))
// CHECK-BE: %[[EXT:[0-9a-zA-Z_.]+]] = extractelement <2 x i64> %{{[0-9a-zA-Z_.]+}}, i32 0
// CHECK-BE: trunc i64 %[[EXT]] to i32
// CHECK-P10-LE: call zeroext i32 @vec_extractm(unsigned int vector[4])(<4 x i32> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK-P10-BE: call zeroext i32 @vec_extractm(unsigned int vector[4])(<4 x i32> noundef %{{[0-9a-zA-Z_.]+}})

void __attribute__((noinline))
test_alt_name_move() {
  i = _m_pmovmskb(ms[0]);
  _m_maskmovq(ms[0], ms[1], (char *)&res64);
}

// CHECK-LABEL: @test_alt_name_move

// CHECK-LABEL: define available_externally signext i32 @_m_pmovmskb
// CHECK: call signext i32 @_mm_movemask_pi8

// CHECK-LABEL: define available_externally void @_m_maskmovq
// CHECK: call void @_mm_maskmove_si64

void __attribute__((noinline))
test_mul() {
  res = _mm_mul_ps(m1, m2);
  res = _mm_mul_ss(m1, m2);
  res64 = _mm_mulhi_pu16(ms[0], ms[1]);
  res64 = _m_pmulhuw(ms[0], ms[1]);
}

// CHECK-LABEL: @test_mul

// CHECK-LABEL: define available_externally <4 x float> @_mm_mul_ps
// CHECK: fmul <4 x float>

// CHECK-LABEL: define available_externally <4 x float> @_mm_mul_ss
// CHECK: call <4 x float> @vec_splat(float vector[4], unsigned int)(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, i32 noundef zeroext 0)
// CHECK: call <4 x float> @vec_splat(float vector[4], unsigned int)(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, i32 noundef zeroext 0)
// CHECK: fmul <4 x float>
// CHECK: call <4 x float> @vec_sel(float vector[4], float vector[4], unsigned int vector[4])(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, <4 x float> noundef %{{[0-9a-zA-Z_.]+}}, <4 x i32> noundef <i32 -1, i32 0, i32 0, i32 0>)

// CHECK-LABEL: define available_externally i64 @_mm_mulhi_pu16
// CHECK-LE: store <16 x i8> <i8 2, i8 3, i8 18, i8 19, i8 6, i8 7, i8 22, i8 23, i8 10, i8 11, i8 26, i8 27, i8 14, i8 15, i8 30, i8 31>, ptr %{{[0-9a-zA-Z_.]+}}, align 16
// CHECK-BE: store <16 x i8> <i8 0, i8 1, i8 16, i8 17, i8 4, i8 5, i8 20, i8 21, i8 0, i8 1, i8 16, i8 17, i8 4, i8 5, i8 20, i8 21>, ptr %{{[0-9a-zA-Z_.]+}}, align 16
// CHECK: call <2 x i64> @vec_splats(unsigned long long)
// CHECK: call <2 x i64> @vec_splats(unsigned long long)
// CHECK: call <4 x i32> @vec_vmuleuh(<8 x i16> noundef %{{[0-9a-zA-Z_.]+}}, <8 x i16> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: call <4 x i32> @vec_vmulouh(<8 x i16> noundef %{{[0-9a-zA-Z_.]+}}, <8 x i16> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: call <4 x i32> @vec_perm(unsigned int vector[4], unsigned int vector[4], unsigned char vector[16])
// CHECK: extractelement <2 x i64> %{{[0-9a-zA-Z_.]+}}, i32 0

// CHECK-LABEL: define available_externally i64 @_m_pmulhuw
// CHECK: call i64 @_mm_mulhi_pu16

void __attribute__((noinline))
test_prefetch() {
  _mm_prefetch(ms, _MM_HINT_NTA);
}

// CHECK-LABEL: @test_prefetch

// CHECK-LABEL: define available_externally void @_mm_prefetch
// CHECK: call void @llvm.prefetch.p0(ptr %{{[0-9a-zA-Z_.]+}}, i32 0, i32 3, i32 1)

void __attribute__((noinline))
test_rcp() {
  res = _mm_rcp_ps(m1);
  res = _mm_rcp_ss(m1);
}

// CHECK-LABEL: @test_rcp

// CHECK-LABEL: define available_externally <4 x float> @_mm_rcp_ps
// CHECK: call <4 x float> @vec_re(float vector[4])

// CHECK-LABEL: define available_externally <4 x float> @_mm_rcp_ss
// CHECK: call <4 x float> @vec_splat(float vector[4], unsigned int)
// CHECK: call <4 x float> @_mm_rcp_ps(<4 x float> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: call <4 x float> @vec_sel(float vector[4], float vector[4], unsigned int vector[4])(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, <4 x float> noundef %{{[0-9a-zA-Z_.]+}}, <4 x i32> noundef <i32 -1, i32 0, i32 0, i32 0>)

void __attribute__((noinline))
test_rsqrt() {
  res = _mm_rsqrt_ps(m1);
  res = _mm_rsqrt_ss(m1);
}

// CHECK-LABEL: @test_rsqrt

// CHECK-LABEL: define available_externally <4 x float> @_mm_rsqrt_ps
// CHECK: call <4 x float> @vec_rsqrte(float vector[4])

// CHECK-LABEL: define available_externally <4 x float> @_mm_rsqrt_ss
// CHECK: call <4 x float> @vec_splat(float vector[4], unsigned int)(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, i32 noundef zeroext 0)
// CHECK: call <4 x float> @vec_rsqrte(float vector[4])
// CHECK: call <4 x float> @vec_sel(float vector[4], float vector[4], unsigned int vector[4])(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, <4 x float> noundef %{{[0-9a-zA-Z_.]+}}, <4 x i32> noundef <i32 -1, i32 0, i32 0, i32 0>)

void __attribute__((noinline))
test_sad() {
  res64 = _mm_sad_pu8(ms[0], ms[1]);
  res64 = _m_psadbw(ms[0], ms[1]);
}

// CHECK-LABEL: @test_sad

// CHECK-LABEL: define available_externally i64 @_mm_sad_pu8
// CHECK: call void @llvm.memset.p0.i64(ptr align 8 %{{[0-9a-zA-Z_.]+}}, i8 0, i64 8, i1 false)
// CHECK: insertelement <2 x i64> <i64 0, i64 poison>, i64 %{{[0-9a-zA-Z_.]+}}, i32 1
// CHECK: insertelement <2 x i64> <i64 0, i64 poison>, i64 %{{[0-9a-zA-Z_.]+}}, i32 1
// CHECK: call <16 x i8> @vec_min(unsigned char vector[16], unsigned char vector[16])
// CHECK: call <16 x i8> @vec_max(unsigned char vector[16], unsigned char vector[16])
// CHECK: call <16 x i8> @vec_sub(unsigned char vector[16], unsigned char vector[16])
// CHECK: call <4 x i32> @vec_sum4s(unsigned char vector[16], unsigned int vector[4])(<16 x i8> noundef %{{[0-9a-zA-Z_.]+}}, <4 x i32> noundef zeroinitializer)
// CHECK: call <4 x i32> @vec_sums(<4 x i32> noundef %{{[0-9a-zA-Z_.]+}}, <4 x i32> noundef zeroinitializer)
// CHECK: %[[EXT:[0-9a-zA-Z_.]+]] = extractelement <4 x i32> %{{[0-9a-zA-Z_.]+}}, i32 3
// CHECK: %[[TRUNC:[0-9a-zA-Z_.]+]] = trunc i32 %[[EXT]] to i16
// CHECK: %[[GEP:[0-9a-zA-Z_.]+]] = getelementptr inbounds [4 x i16], ptr %{{[0-9a-zA-Z_.]+}}, i64 0, i64 0
// CHECK: store i16 %[[TRUNC]], ptr %[[GEP]], align 8

// CHECK-LABEL: define available_externally i64 @_m_psadbw
// CHECK: call i64 @_mm_sad_pu8

void __attribute__((noinline))
test_set() {
  res = _mm_set_ps(fs[0], fs[1], fs[2], fs[3]);
  res = _mm_set_ps1(fs[0]);
  res = _mm_set_ss(fs[0]);
  res = _mm_set1_ps(fs[0]);
  res = _mm_setr_ps(fs[0], fs[1], fs[2], fs[3]);
}

// CHECK-LABEL: @test_set

// CHECK-LABEL: define available_externally <4 x float> @_mm_set_ps
// CHECK: %[[VEC:[0-9a-zA-Z_.]+]] = insertelement <4 x float> poison, float %{{[0-9a-zA-Z_.]+}}, i32 0
// CHECK: %[[VEC2:[0-9a-zA-Z_.]+]] = insertelement <4 x float> %[[VEC]], float %{{[0-9a-zA-Z_.]+}}, i32 1
// CHECK: %[[VEC3:[0-9a-zA-Z_.]+]] = insertelement <4 x float> %[[VEC2]], float %{{[0-9a-zA-Z_.]+}}, i32 2
// CHECK: %[[VEC4:[0-9a-zA-Z_.]+]] = insertelement <4 x float> %[[VEC3]], float %{{[0-9a-zA-Z_.]+}}, i32 3
// CHECK: store <4 x float> %[[VEC4]], ptr %{{[0-9a-zA-Z_.]+}}, align 16

// CHECK-LABEL: define available_externally <4 x float> @_mm_set_ps1
// CHECK: call <4 x float> @_mm_set1_ps

// CHECK-LABEL: define available_externally <4 x float> @_mm_set_ss
// CHECK: %[[VEC:[0-9a-zA-Z_.]+]] = insertelement <4 x float> poison, float %{{[0-9a-zA-Z_.]+}}, i32 0
// CHECK: %[[VEC2:[0-9a-zA-Z_.]+]] = insertelement <4 x float> %[[VEC]], float 0.000000e+00, i32 1
// CHECK: %[[VEC3:[0-9a-zA-Z_.]+]] = insertelement <4 x float> %[[VEC2]], float 0.000000e+00, i32 2
// CHECK: %[[VEC4:[0-9a-zA-Z_.]+]] = insertelement <4 x float> %[[VEC3]], float 0.000000e+00, i32 3
// CHECK: store <4 x float> %[[VEC4]], ptr %{{[0-9a-zA-Z_.]+}}, align 16

// CHECK-LABEL: define available_externally <4 x float> @_mm_set1_ps
// CHECK: %[[VEC:[0-9a-zA-Z_.]+]] = insertelement <4 x float> poison, float %{{[0-9a-zA-Z_.]+}}, i32 0
// CHECK: %[[VEC2:[0-9a-zA-Z_.]+]] = insertelement <4 x float> %[[VEC]], float %{{[0-9a-zA-Z_.]+}}, i32 1
// CHECK: %[[VEC3:[0-9a-zA-Z_.]+]] = insertelement <4 x float> %[[VEC2]], float %{{[0-9a-zA-Z_.]+}}, i32 2
// CHECK: %[[VEC4:[0-9a-zA-Z_.]+]] = insertelement <4 x float> %[[VEC3]], float %{{[0-9a-zA-Z_.]+}}, i32 3
// CHECK: store <4 x float> %[[VEC4]], ptr %{{[0-9a-zA-Z_.]+}}, align 16

// CHECK-LABEL: define available_externally <4 x float> @_mm_setr_ps
// CHECK: %[[VEC:[0-9a-zA-Z_.]+]] = insertelement <4 x float> poison, float %{{[0-9a-zA-Z_.]+}}, i32 0
// CHECK: %[[VEC2:[0-9a-zA-Z_.]+]] = insertelement <4 x float> %[[VEC]], float %{{[0-9a-zA-Z_.]+}}, i32 1
// CHECK: %[[VEC3:[0-9a-zA-Z_.]+]] = insertelement <4 x float> %[[VEC2]], float %{{[0-9a-zA-Z_.]+}}, i32 2
// CHECK: %[[VEC4:[0-9a-zA-Z_.]+]] = insertelement <4 x float> %[[VEC3]], float %{{[0-9a-zA-Z_.]+}}, i32 3
// CHECK: store <4 x float> %[[VEC4]], ptr %{{[0-9a-zA-Z_.]+}}, align 16

void __attribute__((noinline))
test_setzero() {
  res = _mm_setzero_ps();
}

// CHECK-LABEL: @test_setzero

// CHECK-LABEL: define available_externally <4 x float> @_mm_setzero_ps
// CHECK: store <4 x float> zeroinitializer, ptr %{{[0-9a-zA-Z_.]+}}, align 16

void __attribute__((noinline))
test_sfence() {
  _mm_sfence();
}

// CHECK-LABEL: @test_sfence

// CHECK-LABEL: define available_externally void @_mm_sfence
// CHECK: fence release

void __attribute__((noinline))
test_shuffle() {
  res64 = _mm_shuffle_pi16(ms[0], i);
  res = _mm_shuffle_ps(m1, m2, i);
  res64 = _m_pshufw(ms[0], i);
}

// CHECK-LABEL: @test_shuffle

// CHECK-LABEL: define available_externally i64 @_mm_shuffle_pi16
// CHECK: %[[AND:[0-9a-zA-Z_.]+]] = and i32 %{{[0-9a-zA-Z_.]+}}, 3
// CHECK: sext i32 %[[AND]] to i64
// CHECK: %[[SHR:[0-9a-zA-Z_.]+]] = ashr i32 %{{[0-9a-zA-Z_.]+}}, 2
// CHECK: %[[AND2:[0-9a-zA-Z_.]+]] = and i32 %[[SHR]], 3
// CHECK: sext i32 %[[AND2]] to i64
// CHECK: %[[SHR2:[0-9a-zA-Z_.]+]] = ashr i32 %{{[0-9a-zA-Z_.]+}}, 4
// CHECK: %[[AND3:[0-9a-zA-Z_.]+]] = and i32 %[[SHR2]], 3
// CHECK: sext i32 %[[AND3]] to i64
// CHECK: %[[SHR3:[0-9a-zA-Z_.]+]] = ashr i32 %{{[0-9a-zA-Z_.]+}}, 6
// CHECK: %[[AND4:[0-9a-zA-Z_.]+]] = and i32 %[[SHR3]], 3
// CHECK: sext i32 %[[AND4]] to i64
// CHECK: getelementptr inbounds [4 x i16], ptr @_mm_shuffle_pi16.__permute_selectors, i64 0, i64 %{{[0-9a-zA-Z_.]+}}
// CHECK-LE: getelementptr inbounds [4 x i16], ptr %{{[0-9a-zA-Z_.]+}}, i64 0, i64 0
// CHECK-BE: getelementptr inbounds [4 x i16], ptr %{{[0-9a-zA-Z_.]+}}, i64 0, i64 3
// CHECK: getelementptr inbounds [4 x i16], ptr @_mm_shuffle_pi16.__permute_selectors, i64 0, i64 %{{[0-9a-zA-Z_.]+}}
// CHECK-LE: getelementptr inbounds [4 x i16], ptr %{{[0-9a-zA-Z_.]+}}, i64 0, i64 1
// CHECK-BE: getelementptr inbounds [4 x i16], ptr %{{[0-9a-zA-Z_.]+}}, i64 0, i64 2
// CHECK: getelementptr inbounds [4 x i16], ptr @_mm_shuffle_pi16.__permute_selectors, i64 0, i64 %{{[0-9a-zA-Z_.]+}}
// CHECK-LE: getelementptr inbounds [4 x i16], ptr %{{[0-9a-zA-Z_.]+}}, i64 0, i64 2
// CHECK-BE: getelementptr inbounds [4 x i16], ptr %{{[0-9a-zA-Z_.]+}}, i64 0, i64 1
// CHECK: getelementptr inbounds [4 x i16], ptr @_mm_shuffle_pi16.__permute_selectors, i64 0, i64 %{{[0-9a-zA-Z_.]+}}
// CHECK-LE: getelementptr inbounds [4 x i16], ptr %{{[0-9a-zA-Z_.]+}}, i64 0, i64 3
// CHECK-BE: getelementptr inbounds [4 x i16], ptr %{{[0-9a-zA-Z_.]+}}, i64 0, i64 0
// CHECK: call <2 x i64> @vec_splats(unsigned long long)
// CHECK: call <2 x i64> @vec_splats(unsigned long long)
// CHECK: call <2 x i64> @vec_perm(unsigned long long vector[2], unsigned long long vector[2], unsigned char vector[16])
// CHECK: extractelement <2 x i64> %{{[0-9a-zA-Z_.]+}}, i32 0

// CHECK-LABEL: define available_externally <4 x float> @_mm_shuffle_ps
// CHECK: %[[AND:[0-9a-zA-Z_.]+]] = and i32 %{{[0-9a-zA-Z_.]+}}, 3
// CHECK: sext i32 %[[AND]] to i64
// CHECK: %[[SHR:[0-9a-zA-Z_.]+]] = ashr i32 %{{[0-9a-zA-Z_.]+}}, 2
// CHECK: %[[AND2:[0-9a-zA-Z_.]+]] = and i32 %[[SHR]], 3
// CHECK: sext i32 %[[AND2]] to i64
// CHECK: %[[SHR2:[0-9a-zA-Z_.]+]] = ashr i32 %{{[0-9a-zA-Z_.]+}}, 4
// CHECK: %[[AND3:[0-9a-zA-Z_.]+]] = and i32 %[[SHR2]], 3
// CHECK: sext i32 %[[AND3]] to i64
// CHECK: %[[SHR3:[0-9a-zA-Z_.]+]] = ashr i32 %{{[0-9a-zA-Z_.]+}}, 6
// CHECK: %[[AND4:[0-9a-zA-Z_.]+]] = and i32 %[[SHR3]], 3
// CHECK: sext i32 %[[AND4]] to i64
// CHECK: getelementptr inbounds [4 x i32], ptr @_mm_shuffle_ps.__permute_selectors, i64 0, i64
// CHECK: insertelement <4 x i32> %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 0
// CHECK: getelementptr inbounds [4 x i32], ptr @_mm_shuffle_ps.__permute_selectors, i64 0, i64
// CHECK: insertelement <4 x i32> %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 1
// CHECK: getelementptr inbounds [4 x i32], ptr @_mm_shuffle_ps.__permute_selectors, i64 0, i64
// CHECK: %[[ADD:[0-9a-zA-Z_.]+]] = add i32 %{{[0-9a-zA-Z_.]+}}, 269488144
// CHECK: insertelement <4 x i32> %{{[0-9a-zA-Z_.]+}}, i32 %[[ADD]], i32 2
// CHECK: getelementptr inbounds [4 x i32], ptr @_mm_shuffle_ps.__permute_selectors, i64 0, i64
// CHECK: %[[ADD2:[0-9a-zA-Z_.]+]] = add i32 %{{[0-9a-zA-Z_.]+}}, 269488144
// CHECK: insertelement <4 x i32> %{{[0-9a-zA-Z_.]+}}, i32 %[[ADD2]], i32 3
// CHECK: call <4 x float> @vec_perm(float vector[4], float vector[4], unsigned char vector[16])

// CHECK-LABEL: define available_externally i64 @_m_pshufw
// CHECK: call i64 @_mm_shuffle_pi16

void __attribute__((noinline))
test_sqrt() {
  res = _mm_sqrt_ps(m1);
  res = _mm_sqrt_ss(m1);
}

// CHECK-LABEL: @test_sqrt

// CHECK-LABEL: define available_externally <4 x float> @_mm_sqrt_ps
// CHECK: call <4 x float> @vec_sqrt(float vector[4])(<4 x float> noundef %{{[0-9a-zA-Z_.]+}})

// CHECK-LABEL: define available_externally <4 x float> @_mm_sqrt_ss
// CHECK: call <4 x float> @vec_splat(float vector[4], unsigned int)(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, i32 noundef zeroext 0)
// CHECK: call <4 x float> @vec_sqrt(float vector[4])
// CHECK: call <4 x float> @vec_sel(float vector[4], float vector[4], unsigned int vector[4])(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, <4 x float> noundef %{{[0-9a-zA-Z_.]+}}, <4 x i32> noundef <i32 -1, i32 0, i32 0, i32 0>)

void __attribute__((noinline))
test_store() {
  _mm_store_ps(fs, m1);
  _mm_store_ps1(fs, m1);
  _mm_store_ss(fs, m1);
  _mm_store1_ps(fs, m1);
  _mm_storeh_pi(ms, m1);
  _mm_storel_pi(ms, m1);
  _mm_storer_ps(fs, m1);
}

// CHECK-LABEL: @test_store

// CHECK-LABEL: define available_externally void @_mm_store_ps
// CHECK: call void @vec_st(float vector[4], long, float vector[4]*)(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, i64 noundef 0, ptr noundef %{{[0-9a-zA-Z_.]+}})

// CHECK-LABEL: define available_externally void @_mm_store_ps1
// CHECK: call void @_mm_store1_ps

// CHECK-LABEL: define available_externally void @_mm_store_ss
// CHECK: %[[VAL:[0-9a-zA-Z_.]+]] = extractelement <4 x float> %{{[0-9a-zA-Z_.]+}}, i32 0
// CHECK: store float %[[VAL]], ptr %{{[0-9a-zA-Z_.]+}}, align 4

// CHECK-LABEL: define available_externally void @_mm_store1_ps
// CHECK: call <4 x float> @vec_splat(float vector[4], unsigned int)(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, i32 noundef zeroext 0)
// CHECK: call void @_mm_store_ps

// CHECK-LABEL: define available_externally void @_mm_storeh_pi
// CHECK: %[[VAL:[0-9a-zA-Z_.]+]] = extractelement <2 x i64> %{{[0-9a-zA-Z_.]+}}, i32 1
// CHECK: store i64 %[[VAL]], ptr %{{[0-9a-zA-Z_.]+}}, align 8

// CHECK-LABEL: define available_externally void @_mm_storel_pi
// CHECK: %[[VAL:[0-9a-zA-Z_.]+]] = extractelement <2 x i64> %{{[0-9a-zA-Z_.]+}}, i32 0
// CHECK: store i64 %[[VAL]], ptr %{{[0-9a-zA-Z_.]+}}, align 8

// CHECK-LABEL: define available_externally void @_mm_storer_ps
// CHECK: call <4 x float> @vec_perm(float vector[4], float vector[4], unsigned char vector[16])(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, <4 x float> noundef %{{[0-9a-zA-Z_.]+}}, <16 x i8> noundef <i8 28, i8 29, i8 30, i8 31, i8 24, i8 25, i8 26, i8 27, i8 20, i8 21, i8 22, i8 23, i8 16, i8 17, i8 18, i8 19>)
// CHECK: call void @_mm_store_ps

void __attribute__((noinline))
test_stream() {
  _mm_stream_pi(&res64, ms[0]);
  _mm_stream_ps(&fs[0], m1);
}

// CHECK-LABEL: @test_stream

// CHECK-LABEL: define available_externally void @_mm_stream_pi
// CHECK: call void asm sideeffect "\09dcbtstt\090,$0", "b,~{memory}"(ptr %{{[0-9a-zA-Z_.]+}})

// CHECK-LABEL: define available_externally void @_mm_stream_ps
// CHECK: call void asm sideeffect "\09dcbtstt\090,$0", "b,~{memory}"(ptr %{{[0-9a-zA-Z_.]+}})
// CHECK: call void @_mm_store_ps

void __attribute__((noinline))
test_sub() {
  res = _mm_sub_ps(m1, m2);
  res = _mm_sub_ss(m1, m2);
}

// CHECK-LABEL: @test_sub

// CHECK-LABEL: define available_externally <4 x float> @_mm_sub_ps
// CHECK: fsub <4 x float>

// CHECK-LABEL: define available_externally <4 x float> @_mm_sub_ss
// CHECK: call <4 x float> @vec_splat(float vector[4], unsigned int)(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, i32 noundef zeroext 0)
// CHECK: call <4 x float> @vec_splat(float vector[4], unsigned int)(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, i32 noundef zeroext 0)
// CHECK: fsub <4 x float>
// CHECK: call <4 x float> @vec_sel(float vector[4], float vector[4], unsigned int vector[4])(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, <4 x float> noundef %{{[0-9a-zA-Z_.]+}}, <4 x i32> noundef <i32 -1, i32 0, i32 0, i32 0>)

void __attribute__((noinline))
test_transpose() {
  __m128 m3, m4;
  _MM_TRANSPOSE4_PS(m1, m2, m3, m4);
}

// CHECK-LABEL: @test_transpose

// CHECK: call <4 x float> @vec_vmrghw(float vector[4], float vector[4])
// CHECK: call <4 x float> @vec_vmrghw(float vector[4], float vector[4])
// CHECK: call <4 x float> @vec_vmrglw(float vector[4], float vector[4])
// CHECK: call <4 x float> @vec_vmrglw(float vector[4], float vector[4])
// CHECK: call <2 x i64> @vec_mergeh(long long vector[2], long long vector[2])
// CHECK: call <2 x i64> @vec_mergel(long long vector[2], long long vector[2])
// CHECK: call <2 x i64> @vec_mergeh(long long vector[2], long long vector[2])
// CHECK: call <2 x i64> @vec_mergel(long long vector[2], long long vector[2])

void __attribute__((noinline))
test_ucomi() {
  i = _mm_ucomieq_ss(m1, m2);
  i = _mm_ucomige_ss(m1, m2);
  i = _mm_ucomigt_ss(m1, m2);
  i = _mm_ucomile_ss(m1, m2);
  i = _mm_ucomilt_ss(m1, m2);
  i = _mm_ucomineq_ss(m1, m2);
}

// CHECK-LABEL: @test_ucomi

// CHECK-LABEL: define available_externally signext i32 @_mm_ucomieq_ss
// CHECK: %[[VAL1:[0-9a-zA-Z_.]+]] = extractelement <4 x float> %{{[0-9a-zA-Z_.]+}}, i32 0
// CHECK: %[[VAL2:[0-9a-zA-Z_.]+]] = extractelement <4 x float> %{{[0-9a-zA-Z_.]+}}, i32 0
// CHECK: fcmp oeq float %[[VAL1]], %[[VAL2]]

// CHECK-LABEL: define available_externally signext i32 @_mm_ucomige_ss
// CHECK: %[[VAL1:[0-9a-zA-Z_.]+]] = extractelement <4 x float> %{{[0-9a-zA-Z_.]+}}, i32 0
// CHECK: %[[VAL2:[0-9a-zA-Z_.]+]] = extractelement <4 x float> %{{[0-9a-zA-Z_.]+}}, i32 0
// CHECK: fcmp oge float %[[VAL1]], %[[VAL2]]

// CHECK-LABEL: define available_externally signext i32 @_mm_ucomigt_ss
// CHECK: %[[VAL1:[0-9a-zA-Z_.]+]] = extractelement <4 x float> %{{[0-9a-zA-Z_.]+}}, i32 0
// CHECK: %[[VAL2:[0-9a-zA-Z_.]+]] = extractelement <4 x float> %{{[0-9a-zA-Z_.]+}}, i32 0
// CHECK: fcmp ogt float %[[VAL1]], %[[VAL2]]

// CHECK-LABEL: define available_externally signext i32 @_mm_ucomile_ss
// CHECK: %[[VAL1:[0-9a-zA-Z_.]+]] = extractelement <4 x float> %{{[0-9a-zA-Z_.]+}}, i32 0
// CHECK: %[[VAL2:[0-9a-zA-Z_.]+]] = extractelement <4 x float> %{{[0-9a-zA-Z_.]+}}, i32 0
// CHECK: fcmp ole float %[[VAL1]], %[[VAL2]]

// CHECK-LABEL: define available_externally signext i32 @_mm_ucomilt_ss
// CHECK: %[[VAL1:[0-9a-zA-Z_.]+]] = extractelement <4 x float> %{{[0-9a-zA-Z_.]+}}, i32 0
// CHECK: %[[VAL2:[0-9a-zA-Z_.]+]] = extractelement <4 x float> %{{[0-9a-zA-Z_.]+}}, i32 0
// CHECK: fcmp olt float %[[VAL1]], %[[VAL2]]

// CHECK-LABEL: define available_externally signext i32 @_mm_ucomineq_ss
// CHECK: %[[VAL1:[0-9a-zA-Z_.]+]] = extractelement <4 x float> %{{[0-9a-zA-Z_.]+}}, i32 0
// CHECK: %[[VAL2:[0-9a-zA-Z_.]+]] = extractelement <4 x float> %{{[0-9a-zA-Z_.]+}}, i32 0
// CHECK: fcmp une float %[[VAL1]], %[[VAL2]]

void __attribute__((noinline))
test_undefined() {
  res = _mm_undefined_ps();
}

// CHECK-LABEL: @test_undefined

// CHECK-LABEL: define available_externally <4 x float> @_mm_undefined_ps
// CHECK: alloca <4 x float>, align 16
// CHECK: load <4 x float>, ptr %[[ADDR:[0-9a-zA-Z_.]+]], align 16
// CHECK: load <4 x float>, ptr %[[ADDR]], align 16

void __attribute__((noinline))
test_unpack() {
  res = _mm_unpackhi_ps(m1, m2);
  res = _mm_unpacklo_ps(m1, m2);
}

// CHECK-LABEL: @test_unpack

// CHECK-LABEL: define available_externally <4 x float> @_mm_unpackhi_ps
// CHECK: call <4 x float> @vec_vmrglw(float vector[4], float vector[4])

// CHECK-LABEL: define available_externally <4 x float> @_mm_unpacklo_ps
// CHECK: call <4 x float> @vec_vmrghw(float vector[4], float vector[4])
