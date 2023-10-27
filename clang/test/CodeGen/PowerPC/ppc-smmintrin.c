// REQUIRES: powerpc-registered-target

// RUN: %clang -S -emit-llvm -target powerpc64le-unknown-linux-gnu -mcpu=pwr8 -ffreestanding -DNO_WARN_X86_INTRINSICS %s \
// RUN:   -fno-discard-value-names -mllvm -disable-llvm-optzns -o - | llvm-cxxfilt -n | FileCheck %s
// RUN: %clang -S -emit-llvm -target powerpc64-unknown-linux-gnu -mcpu=pwr8 -ffreestanding -DNO_WARN_X86_INTRINSICS %s \
// RUN:   -fno-discard-value-names -mllvm -disable-llvm-optzns -o - | llvm-cxxfilt -n | FileCheck %s

// RUN: %clang -S -emit-llvm -target powerpc64le-unknown-freebsd13.0 -mcpu=pwr8 -ffreestanding -DNO_WARN_X86_INTRINSICS %s \
// RUN:   -fno-discard-value-names -mllvm -disable-llvm-optzns -o - | llvm-cxxfilt -n | FileCheck %s
// RUN: %clang -S -emit-llvm -target powerpc64-unknown-freebsd13.0 -mcpu=pwr8 -ffreestanding -DNO_WARN_X86_INTRINSICS %s \
// RUN:   -fno-discard-value-names -mllvm -disable-llvm-optzns -o - | llvm-cxxfilt -n | FileCheck %s

// RUN: %clang -S -emit-llvm -target powerpc64le-unknown-linux-gnu -mcpu=pwr10 -ffreestanding -DNO_WARN_X86_INTRINSICS %s \
// RUN:   -fno-discard-value-names -mllvm -disable-llvm-optzns -o - | llvm-cxxfilt -n | FileCheck %s --check-prefix=P10
// RUN: %clang -S -emit-llvm -target powerpc64-unknown-linux-gnu -mcpu=pwr10 -ffreestanding -DNO_WARN_X86_INTRINSICS %s \
// RUN:   -fno-discard-value-names -mllvm -disable-llvm-optzns -o - | llvm-cxxfilt -n | FileCheck %s --check-prefix=P10

// RUN: %clang -x c++ -S -emit-llvm -target powerpc64le-unknown-linux-gnu -mcpu=pwr8 -ffreestanding -DNO_WARN_X86_INTRINSICS %s \
// RUN:   -fno-discard-value-names -mllvm -disable-llvm-optzns -fsyntax-only
// RUN: %clang -x c++ -S -emit-llvm -target powerpc64le-unknown-linux-gnu -mcpu=pwr10 -ffreestanding -DNO_WARN_X86_INTRINSICS %s \
// RUN:   -fno-discard-value-names -mllvm -disable-llvm-optzns -fsyntax-only

// RUN: %clang -S -emit-llvm -target powerpc64le-unknown-freebsd13.0 -mcpu=pwr8 -ffreestanding -DNO_WARN_X86_INTRINSICS %s \
// RUN:   -fno-discard-value-names -mllvm -disable-llvm-optzns -o - | llvm-cxxfilt -n | FileCheck %s
// RUN: %clang -S -emit-llvm -target powerpc64-unknown-freebsd13.0 -mcpu=pwr8 -ffreestanding -DNO_WARN_X86_INTRINSICS %s \
// RUN:   -fno-discard-value-names -mllvm -disable-llvm-optzns -o - | llvm-cxxfilt -n | FileCheck %s

// RUN: %clang -S -emit-llvm -target powerpc64-ibm-aix -mcpu=pwr10 -ffreestanding -DNO_WARN_X86_INTRINSICS %s \
// RUN:   -fno-discard-value-names -mllvm -disable-llvm-optzns -o - | llvm-cxxfilt -n | FileCheck %s --check-prefix=P10
// RUN: %clang -S -emit-llvm -target powerpc64-ibm-aix -mcpu=pwr8 -ffreestanding -DNO_WARN_X86_INTRINSICS %s \
// RUN:   -fno-discard-value-names -mllvm -disable-llvm-optzns -o - | llvm-cxxfilt -n | FileCheck %s

#include <smmintrin.h>

__m128 mn1, mn2;
__m128d md1, md2;
__m128i mi, m1, m2;

void __attribute__((noinline))
test_extract() {
  _mm_extract_epi8(mi, 0);
  _mm_extract_epi32(mi, 0);
  _mm_extract_epi64(mi, 0);
  _mm_extract_ps((__m128)mi, 0);
}

// CHECK-LABEL: @test_extract

// CHECK-LABEL: define available_externally signext i32 @_mm_extract_epi8(<2 x i64> noundef %{{[0-9a-zA-Z_.]+}}, i32 noundef signext %{{[0-9a-zA-Z_.]+}})
// CHECK: %[[AND:[0-9a-zA-Z_.]+]] = and i32 %{{[0-9a-zA-Z_.]+}}, 15
// CHECK: %[[EXT:[0-9a-zA-Z_.]+]] = extractelement <16 x i8> %{{[0-9a-zA-Z_.]+}}, i32 %[[AND]]
// CHECK: zext i8 %[[EXT]] to i32

// CHECK-LABEL: define available_externally signext i32 @_mm_extract_epi32(<2 x i64> noundef %{{[0-9a-zA-Z_.]+}}, i32 noundef signext %{{[0-9a-zA-Z_.]+}})
// CHECK: %[[AND:[0-9a-zA-Z_.]+]] = and i32 %{{[0-9a-zA-Z_.]+}}, 3
// CHECK: extractelement <4 x i32> %{{[0-9a-zA-Z_.]+}}, i32 %[[AND]]

// CHECK-LABEL: define available_externally signext i32 @_mm_extract_epi64(<2 x i64> noundef %{{[0-9a-zA-Z_.]+}}, i32 noundef signext %{{[0-9a-zA-Z_.]+}})
// CHECK: %[[AND:[0-9a-zA-Z_.]+]] = and i32 %{{[0-9a-zA-Z_.]+}}, 1
// CHECK: %[[EXT:[0-9a-zA-Z_.]+]] = extractelement <2 x i64> %{{[0-9a-zA-Z_.]+}}, i32 %[[AND]]
// CHECK: trunc i64 %[[EXT]] to i32

// CHECK-LABEL: define available_externally signext i32 @_mm_extract_ps(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, i32 noundef signext %{{[0-9a-zA-Z_.]+}})
// CHECK: %[[AND:[0-9a-zA-Z_.]+]] = and i32 %{{[0-9a-zA-Z_.]+}}, 3
// CHECK: extractelement <4 x i32> %{{[0-9a-zA-Z_.]+}}, i32 %[[AND]]

void __attribute__((noinline))
test_blend() {
  _mm_blend_epi16(m1, m2, 0);
  _mm_blendv_epi8(m1, m2, mi);
  _mm_blend_ps(mn1, mn2, 0);
  _mm_blendv_ps(mn1, mn2, mn1);
  _mm_blend_pd(md1, md2, 0);
  _mm_blendv_pd(md1, md2, md1);
}

// CHECK-LABEL: @test_blend

// CHECK-LABEL: define available_externally <2 x i64> @_mm_blend_epi16(<2 x i64> noundef %{{[0-9a-zA-Z_.]+}}, <2 x i64> noundef %{{[0-9a-zA-Z_.]+}}, i32 noundef signext %{{[0-9a-zA-Z_.]+}})
// CHECK: %[[TRUNC:[0-9a-zA-Z_.]+]] = trunc i32 %{{[0-9a-zA-Z_.]+}} to i8
// CHECK: call <16 x i8> @vec_splats(unsigned char)(i8 noundef zeroext %[[TRUNC]])
// CHECK: call <16 x i8> @llvm.ppc.altivec.vgbbd(<16 x i8> %{{[0-9a-zA-Z_.]+}})
// CHECK: %[[PACK:[0-9a-zA-Z_.]+]] = call <8 x i16> @vec_unpackh(signed char vector[16])
// CHECK: store <8 x i16> %[[PACK]], ptr %{{[0-9a-zA-Z_.]+}}, align 16
// BE: %[[REVE:[0-9a-zA-Z_.]+]] = call <8 x i16> @vec_reve(unsigned short vector[8])
// BE: store <8 x i16> %[[REVE]], ptr %{{[0-9a-zA-Z_.]+}}, align 16
// CHECK: call <8 x i16> @vec_sel(unsigned short vector[8], unsigned short vector[8], unsigned short vector[8])

// P10-LABEL: define available_externally <2 x i64> @_mm_blendv_epi8(<2 x i64> noundef %{{[0-9a-zA-Z_.]+}}, <2 x i64> noundef %{{[0-9a-zA-Z_.]+}}, <2 x i64> noundef %{{[0-9a-zA-Z_.]+}})
// P10: call <16 x i8> @vec_blendv(signed char vector[16], signed char vector[16], unsigned char vector[16])(<16 x i8> noundef %{{[0-9a-zA-Z_.]+}}, <16 x i8> noundef %{{[0-9a-zA-Z_.]+}}, <16 x i8> noundef %{{[0-9a-zA-Z_.]+}})

// CHECK-LABEL: define available_externally <2 x i64> @_mm_blendv_epi8(<2 x i64> noundef %{{[0-9a-zA-Z_.]+}}, <2 x i64> noundef %{{[0-9a-zA-Z_.]+}}, <2 x i64> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: call <16 x i8> @vec_splats(unsigned char)(i8 noundef zeroext 7)
// CHECK: call <16 x i8> @vec_sra(unsigned char vector[16], unsigned char vector[16])
// CHECK: call <16 x i8> @vec_sel(signed char vector[16], signed char vector[16], unsigned char vector[16])

void __attribute__((noinline))
test_insert() {
  _mm_insert_epi8(m1, 1, 0);
  _mm_insert_epi32(m1, 1, 0);
  _mm_insert_epi64(m1, 0xFFFFFFFF1L, 0);
}

// CHECK-LABEL: @test_insert

// CHECK-LABEL: define available_externally <2 x i64> @_mm_insert_epi8(<2 x i64> noundef %{{[0-9a-zA-Z_.]+}}, i32 noundef signext %{{[0-9a-zA-Z_.]+}}, i32 noundef signext %{{[0-9a-zA-Z_.]+}})
// CHECK: %[[TRUNC:[0-9a-zA-Z_.]+]] = trunc i32 %{{[0-9a-zA-Z_.]+}} to i8
// CHECK: %[[AND:[0-9a-zA-Z_.]+]] = and i32 %{{[0-9a-zA-Z_.]+}}, 15
// CHECK: insertelement <16 x i8> %{{[0-9a-zA-Z_.]+}}, i8 %[[TRUNC]], i32 %[[AND]]

// CHECK-LABEL: define available_externally <2 x i64> @_mm_insert_epi32(<2 x i64> noundef %{{[0-9a-zA-Z_.]+}}, i32 noundef signext %{{[0-9a-zA-Z_.]+}}, i32 noundef signext %{{[0-9a-zA-Z_.]+}})
// CHECK: %[[AND:[0-9a-zA-Z_.]+]] = and i32 %{{[0-9a-zA-Z_.]+}}, 3
// CHECK: insertelement <4 x i32> %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, i32 %[[AND]]

// CHECK-LABEL: define available_externally <2 x i64> @_mm_insert_epi64(<2 x i64> noundef %{{[0-9a-zA-Z_.]+}}, i64 noundef %{{[0-9a-zA-Z_.]+}}, i32 noundef signext %{{[0-9a-zA-Z_.]+}})
// CHECK: %[[AND:[0-9a-zA-Z_.]+]] = and i32 %{{[0-9a-zA-Z_.]+}}, 1
// CHECK: insertelement <2 x i64> %{{[0-9a-zA-Z_.]+}}, i64 %{{[0-9a-zA-Z_.]+}}, i32 %[[AND:[0-9a-zA-Z_.]+]]

void __attribute__((noinline))
test_convert() {
  _mm_cvtepi16_epi32(m1);
  _mm_cvtepi16_epi64(m1);
  _mm_cvtepi32_epi64(m1);
  _mm_cvtepi8_epi16(m1);
  _mm_cvtepi8_epi32(m1);
  _mm_cvtepi8_epi64(m1);
  _mm_cvtepu16_epi32(m1);
  _mm_cvtepu16_epi64(m1);
  _mm_cvtepu32_epi64(m1);
  _mm_cvtepu8_epi16(m1);
  _mm_cvtepu8_epi32(m1);
  _mm_cvtepu8_epi64(m1);
}

// CHECK-LABEL: @test_convert

// CHECK-LABEL: define available_externally <2 x i64> @_mm_cvtepi16_epi32(<2 x i64> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: call <4 x i32> @vec_unpackh(short vector[8])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_cvtepi16_epi64(<2 x i64> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: call <4 x i32> @vec_unpackh(short vector[8])
// CHECK: call <2 x i64> @vec_unpackh(int vector[4])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_cvtepi32_epi64(<2 x i64> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: call <2 x i64> @vec_unpackh(int vector[4])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_cvtepi8_epi16(<2 x i64> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: call <8 x i16> @vec_unpackh(signed char vector[16])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_cvtepi8_epi32(<2 x i64> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: call <8 x i16> @vec_unpackh(signed char vector[16])
// CHECK: call <4 x i32> @vec_unpackh(short vector[8])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_cvtepi8_epi64(<2 x i64> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: call <8 x i16> @vec_unpackh(signed char vector[16])
// CHECK: call <4 x i32> @vec_unpackh(short vector[8])
// CHECK: call <2 x i64> @vec_unpackh(int vector[4])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_cvtepu16_epi32(<2 x i64> noundef %{{[0-9a-zA-Z_.]+}})
// LE: call <8 x i16> @vec_mergeh(unsigned short vector[8], unsigned short vector[8])(<8 x i16> noundef %{{[0-9a-zA-Z_.]+}}, <8 x i16> noundef zeroinitializer)
// BE: call <8 x i16> @vec_mergeh(unsigned short vector[8], unsigned short vector[8])(<8 x i16> noundef zeroinitializer, <8 x i16> noundef %{{[0-9a-zA-Z_.]+}})

// CHECK-LABEL: define available_externally <2 x i64> @_mm_cvtepu16_epi64(<2 x i64> noundef %{{[0-9a-zA-Z_.]+}})
// LE: call <8 x i16> @vec_mergeh(unsigned short vector[8], unsigned short vector[8])(<8 x i16> noundef %{{[0-9a-zA-Z_.]+}}, <8 x i16> noundef zeroinitializer)
// LE: call <4 x i32> @vec_mergeh(unsigned int vector[4], unsigned int vector[4])(<4 x i32> noundef %{{[0-9a-zA-Z_.]+}}, <4 x i32> noundef zeroinitializer)
// BE: call <8 x i16> @vec_mergeh(unsigned short vector[8], unsigned short vector[8])(<8 x i16> noundef zeroinitializer, <8 x i16> noundef %{{[0-9a-zA-Z_.]+}})
// BE: call <4 x i32> @vec_mergeh(unsigned int vector[4], unsigned int vector[4])(<4 x i32> noundef zeroinitializer, <4 x i32> noundef %{{[0-9a-zA-Z_.]+}})

// CHECK-LABEL: define available_externally <2 x i64> @_mm_cvtepu32_epi64(<2 x i64> noundef %{{[0-9a-zA-Z_.]+}})
// LE: call <4 x i32> @vec_mergeh(unsigned int vector[4], unsigned int vector[4])(<4 x i32> noundef %{{[0-9a-zA-Z_.]+}}, <4 x i32> noundef zeroinitializer)
// BE: call <4 x i32> @vec_mergeh(unsigned int vector[4], unsigned int vector[4])(<4 x i32> noundef zeroinitializer, <4 x i32> noundef %{{[0-9a-zA-Z_.]+}})

// CHECK-LABEL: define available_externally <2 x i64> @_mm_cvtepu8_epi16(<2 x i64> noundef %{{[0-9a-zA-Z_.]+}})
// LE: call <16 x i8> @vec_mergeh(unsigned char vector[16], unsigned char vector[16])(<16 x i8> noundef %{{[0-9a-zA-Z_.]+}}, <16 x i8> noundef zeroinitializer)
// BE: call <16 x i8> @vec_mergeh(unsigned char vector[16], unsigned char vector[16])(<16 x i8> noundef zeroinitializer, <16 x i8> noundef %{{[0-9a-zA-Z_.]+}})

// CHECK-LABEL: define available_externally <2 x i64> @_mm_cvtepu8_epi32(<2 x i64> noundef %{{[0-9a-zA-Z_.]+}})
// LE: call <16 x i8> @vec_mergeh(unsigned char vector[16], unsigned char vector[16])(<16 x i8> noundef %{{[0-9a-zA-Z_.]+}}, <16 x i8> noundef zeroinitializer)
// LE: call <8 x i16> @vec_mergeh(unsigned short vector[8], unsigned short vector[8])(<8 x i16> noundef %{{[0-9a-zA-Z_.]+}}, <8 x i16> noundef zeroinitializer)
// BE: call <16 x i8> @vec_mergeh(unsigned char vector[16], unsigned char vector[16])(<16 x i8> noundef zeroinitializer, <16 x i8> noundef %{{[0-9a-zA-Z_.]+}})
// BE: call <8 x i16> @vec_mergeh(unsigned short vector[8], unsigned short vector[8])(<8 x i16> noundef zeroinitializer, <8 x i16> noundef %{{[0-9a-zA-Z_.]+}})

// CHECK-LABEL: define available_externally <2 x i64> @_mm_cvtepu8_epi64(<2 x i64> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: call <16 x i8> @vec_mergeh(unsigned char vector[16], unsigned char vector[16])
// CHECK: call <8 x i16> @vec_mergeh(unsigned short vector[8], unsigned short vector[8])
// CHECK: call <4 x i32> @vec_mergeh(unsigned int vector[4], unsigned int vector[4])

void __attribute__((noinline))
test_minmax() {
  _mm_max_epi32(m1, m2);
  _mm_max_epi8(m1, m2);
  _mm_max_epu16(m1, m2);
  _mm_max_epu32(m1, m2);
  _mm_min_epi32(m1, m2);
  _mm_min_epi8(m1, m2);
  _mm_min_epu16(m1, m2);
  _mm_min_epu32(m1, m2);
  _mm_minpos_epu16(m1);
}

// CHECK-LABEL: @test_minmax

// CHECK-LABEL: define available_externally <2 x i64> @_mm_max_epi32(<2 x i64> noundef %{{[0-9a-zA-Z_.]+}}, <2 x i64> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: call <4 x i32> @vec_max(int vector[4], int vector[4])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_max_epi8(<2 x i64> noundef %{{[0-9a-zA-Z_.]+}}, <2 x i64> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: call <16 x i8> @vec_max(signed char vector[16], signed char vector[16])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_max_epu16(<2 x i64> noundef %{{[0-9a-zA-Z_.]+}}, <2 x i64> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: call <8 x i16> @vec_max(unsigned short vector[8], unsigned short vector[8])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_max_epu32(<2 x i64> noundef %{{[0-9a-zA-Z_.]+}}, <2 x i64> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: call <4 x i32> @vec_max(unsigned int vector[4], unsigned int vector[4])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_min_epi32(<2 x i64> noundef %{{[0-9a-zA-Z_.]+}}, <2 x i64> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: call <4 x i32> @vec_min(int vector[4], int vector[4])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_min_epi8(<2 x i64> noundef %{{[0-9a-zA-Z_.]+}}, <2 x i64> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: call <16 x i8> @vec_min(signed char vector[16], signed char vector[16])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_min_epu16(<2 x i64> noundef %{{[0-9a-zA-Z_.]+}}, <2 x i64> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: call <8 x i16> @vec_min(unsigned short vector[8], unsigned short vector[8])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_min_epu32(<2 x i64> noundef %{{[0-9a-zA-Z_.]+}}, <2 x i64> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: call <4 x i32> @vec_min(unsigned int vector[4], unsigned int vector[4])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_minpos_epu16(<2 x i64> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: call void @llvm.memset.p0.i64(ptr align 16 %{{[0-9a-zA-Z_.]+}}, i8 0, i64 16, i1 false)
// CHECK: extractelement <8 x i16> %{{[0-9a-zA-Z_.]+}}, i16 %{{[0-9a-zA-Z_.]+}}
// CHECK: %[[VEXT:[0-9a-zA-Z_.]+]] = extractelement <8 x i16> %{{[0-9a-zA-Z_.]+}}, i64 %{{[0-9a-zA-Z_.]+}}
// CHECK: zext i16 %[[VEXT]] to i32
// CHECK: zext i16 %{{[0-9a-zA-Z_.]+}} to i32
// CHECK: extractelement <8 x i16> %{{[0-9a-zA-Z_.]+}}, i64 %{{[0-9a-zA-Z_.]+}}
// CHECK: add i64 %{{[0-9a-zA-Z_.]+}}, 1

void __attribute__((noinline))
test_round() {
  _mm_round_ps(mn1, 0);
  _mm_round_ss(mn1, mn2, 0);
  _mm_round_pd(md1, 0);
  _mm_round_sd(md1, md2, 0);
}

// CHECK-LABEL: @test_round

// CHECK-LABEL: define available_externally <4 x float> @_mm_round_ps(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, i32 noundef signext %{{[0-9a-zA-Z_.]+}})
// CHECK: call double @llvm.ppc.readflm()
// CHECK: call void @llvm.ppc.mtfsf(i32 3, double %{{[0-9a-zA-Z_.]+}})
// CHECK: %{{[0-9a-zA-Z_.]+}} = call <4 x float> asm "", "=^wa,0"
// CHECK: call double @llvm.ppc.readflm()
// P10: call double @llvm.ppc.mffsl()
// CHECK: call double @llvm.ppc.setrnd(i32 0)
// CHECK: %{{[0-9a-zA-Z_.]+}} = call <4 x float> asm "", "=^wa,0"
// CHECK: call <4 x float> @vec_rint(float vector[4])
// CHECK: call void asm sideeffect "", "^wa"
// CHECK: call double @llvm.ppc.setrnd(i32 %{{[0-9a-zA-Z_.]+}})
// CHECK: call <4 x float> @vec_floor(float vector[4])
// CHECK: call <4 x float> @vec_ceil(float vector[4])
// CHECK: call <4 x float> @vec_trunc(float vector[4])
// CHECK: call <4 x float> @vec_rint(float vector[4])
// CHECK: call void asm sideeffect "", "^wa"
// CHECK: call double @llvm.ppc.readflm()
// P10: call double @llvm.ppc.mffsl()
// CHECK: call void @llvm.ppc.mtfsf(i32 3, double %{{[0-9a-zA-Z_.]+}})

// CHECK-LABEL: define available_externally <4 x float> @_mm_round_ss(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, <4 x float> noundef %{{[0-9a-zA-Z_.]+}}, i32 noundef signext %{{[0-9a-zA-Z_.]+}})
// CHECK: call <4 x float> @_mm_round_ps(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, i32 noundef signext %{{[0-9a-zA-Z_.]+}})
// CHECK: extractelement <4 x float> %{{[0-9a-zA-Z_.]+}}, i32 0

// CHECK-LABEL: define available_externally <2 x double> @_mm_round_pd(<2 x double> noundef %{{[0-9a-zA-Z_.]+}}, i32 noundef signext %{{[0-9a-zA-Z_.]+}})
// CHECK: call double @llvm.ppc.readflm()
// CHECK: call void @llvm.ppc.mtfsf(i32 3, double %{{[0-9a-zA-Z_.]+}})
// CHECK: %{{[0-9a-zA-Z_.]+}} = call <2 x double> asm "", "=^wa,0"
// CHECK: call double @llvm.ppc.readflm()
// P10: call double @llvm.ppc.mffsl()
// CHECK: call double @llvm.ppc.setrnd(i32 0)
// CHECK: %{{[0-9a-zA-Z_.]+}} = call <2 x double> asm "", "=^wa,0"
// CHECK: call <2 x double> @vec_rint(double vector[2])
// CHECK: call void asm sideeffect "", "^wa"
// CHECK: call double @llvm.ppc.setrnd(i32 %{{[0-9a-zA-Z_.]+}})
// CHECK: call <2 x double> @vec_floor(double vector[2])
// CHECK: call <2 x double> @vec_ceil(double vector[2])
// CHECK: call <2 x double> @vec_trunc(double vector[2])
// CHECK: call <2 x double> @vec_rint(double vector[2])
// CHECK: call void asm sideeffect "", "^wa"
// CHECK: call double @llvm.ppc.readflm()
// P10: call double @llvm.ppc.mffsl()
// CHECK: call void @llvm.ppc.mtfsf(i32 3, double %{{[0-9a-zA-Z_.]+}})

// CHECK-LABEL: define available_externally <2 x double> @_mm_round_sd(<2 x double> noundef %{{[0-9a-zA-Z_.]+}}, <2 x double> noundef %{{[0-9a-zA-Z_.]+}}, i32 noundef signext %{{[0-9a-zA-Z_.]+}})
// CHECK: call <2 x double> @_mm_round_pd(<2 x double> noundef %{{[0-9a-zA-Z_.]+}}, i32 noundef signext %{{[0-9a-zA-Z_.]+}})
// CHECK: extractelement <2 x double> %{{[0-9a-zA-Z_.]+}}, i32 0
// CHECK: extractelement <2 x double> %{{[0-9a-zA-Z_.]+}}, i32 1

void __attribute__((noinline))
test_testing() {
  _mm_testc_si128(m1, m2);
  _mm_testnzc_si128(m1, m2);
  _mm_testz_si128(m1, m2);
}

// CHECK-LABEL: @test_testing

// CHECK-LABEL: define available_externally signext i32 @_mm_testc_si128(<2 x i64> noundef %{{[0-9a-zA-Z_.]+}}, <2 x i64> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: call <16 x i8> @vec_nor(unsigned char vector[16], unsigned char vector[16])
// CHECK: call <16 x i8> @vec_and(unsigned char vector[16], unsigned char vector[16])
// CHECK: call signext i32 @vec_all_eq(unsigned char vector[16], unsigned char vector[16])(<16 x i8> noundef %call1, <16 x i8> noundef zeroinitializer)

// CHECK-LABEL: define available_externally signext i32 @_mm_testnzc_si128(<2 x i64> noundef %{{[0-9a-zA-Z_.]+}}, <2 x i64> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: call signext i32 @_mm_testz_si128(<2 x i64> noundef %{{[0-9a-zA-Z_.]+}}, <2 x i64> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: call signext i32 @_mm_testc_si128(<2 x i64> noundef %{{[0-9a-zA-Z_.]+}}, <2 x i64> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: zext i1 %{{[0-9a-zA-Z_.]+}} to i32

// CHECK-LABEL: define available_externally signext i32 @_mm_testz_si128(<2 x i64> noundef %{{[0-9a-zA-Z_.]+}}, <2 x i64> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: call <16 x i8> @vec_and(unsigned char vector[16], unsigned char vector[16])
// CHECK: call signext i32 @vec_all_eq(unsigned char vector[16], unsigned char vector[16])(<16 x i8> noundef %call, <16 x i8> noundef zeroinitializer)

void __attribute__((noinline))
test_compare() {
  _mm_cmpeq_epi64(m1, m2);
  _mm_cmpgt_epi64(m1, m2);
}

// CHECK-LABEL: @test_compare

// CHECK-LABEL: define available_externally <2 x i64> @_mm_cmpeq_epi64(<2 x i64> noundef %{{[0-9a-zA-Z_.]+}}, <2 x i64> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: call <2 x i64> @vec_cmpeq(long long vector[2], long long vector[2])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_cmpgt_epi64(<2 x i64> noundef %{{[0-9a-zA-Z_.]+}}, <2 x i64> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: call <2 x i64> @vec_cmpgt(long long vector[2], long long vector[2])

void __attribute__((noinline))
test_mul() {
  _mm_mul_epi32(m1, m2);
  _mm_mullo_epi32(m1, m2);
}

// CHECK-LABEL: @test_mul

// CHECK-LABEL: define available_externally <2 x i64> @_mm_mul_epi32(<2 x i64> noundef %{{[0-9a-zA-Z_.]+}}, <2 x i64> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: %call = call <2 x i64> @vec_mule(int vector[4], int vector[4])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_mullo_epi32(<2 x i64> noundef %{{[0-9a-zA-Z_.]+}}, <2 x i64> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: %call = call <4 x i32> @vec_mul(unsigned int vector[4], unsigned int vector[4])

void __attribute__((noinline))
test_packus() {
  _mm_packus_epi32(m1, m2);
}

// CHECK-LABEL: @test_packus

// CHECK-LABEL: define available_externally <2 x i64> @_mm_packus_epi32(<2 x i64> noundef %{{[0-9a-zA-Z_.]+}}, <2 x i64> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: call <8 x i16> @vec_packsu(int vector[4], int vector[4])(<4 x i32> noundef %1, <4 x i32> noundef %3)

// To test smmintrin.h includes tmmintrin.h

void __attribute__((noinline))
test_abs_ssse3() {
  _mm_abs_epi16(m1);
}

// CHECK-LABEL: @test_abs_ssse3

// CHECK-LABEL: define available_externally <2 x i64> @_mm_abs_epi16(<2 x i64> noundef %{{[0-9a-zA-Z_.]+}})
