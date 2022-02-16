// REQUIRES: powerpc-registered-target

// RUN: %clang -S -emit-llvm -target powerpc64le-unknown-linux-gnu -mcpu=pwr8 -ffreestanding -DNO_WARN_X86_INTRINSICS %s \
// RUN:   -fno-discard-value-names -mllvm -disable-llvm-optzns -o - | llvm-cxxfilt -n | FileCheck %s
// RUN: %clang -S -emit-llvm -target powerpc64-unknown-linux-gnu -mcpu=pwr8 -ffreestanding -DNO_WARN_X86_INTRINSICS %s \
// RUN:   -fno-discard-value-names -mllvm -disable-llvm-optzns -o - | llvm-cxxfilt -n | FileCheck %s

// RUN: %clang -S -emit-llvm -target powerpc64le-unknown-freebsd13.0 -mcpu=pwr8 -ffreestanding -DNO_WARN_X86_INTRINSICS %s \
// RUN:   -fno-discard-value-names -mllvm -disable-llvm-optzns -o - | llvm-cxxfilt -n | FileCheck %s
// RUN: %clang -S -emit-llvm -target powerpc64-unknown-freebsd13.0 -mcpu=pwr8 -ffreestanding -DNO_WARN_X86_INTRINSICS %s \
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
}

// CHECK-LABEL: @test_blend

// CHECK-LABEL: define available_externally <2 x i64> @_mm_blend_epi16(<2 x i64> noundef %{{[0-9a-zA-Z_.]+}}, <2 x i64> noundef %{{[0-9a-zA-Z_.]+}}, i32 noundef signext %{{[0-9a-zA-Z_.]+}})
// CHECK: %[[TRUNC:[0-9a-zA-Z_.]+]] = trunc i32 %{{[0-9a-zA-Z_.]+}} to i8
// CHECK: call <16 x i8> @vec_splats(signed char)(i8 noundef signext %[[TRUNC]])
// CHECK: call <16 x i8> @llvm.ppc.altivec.vgbbd(<16 x i8> %{{[0-9a-zA-Z_.]+}})
// CHECK: %[[PACK:[0-9a-zA-Z_.]+]] = call <8 x i16> @vec_unpackh(signed char vector[16])
// CHECK: store <8 x i16> %[[PACK]], <8 x i16>* %{{[0-9a-zA-Z_.]+}}, align 16
// BE: %[[REVE:[0-9a-zA-Z_.]+]] = call <8 x i16> @vec_reve(unsigned short vector[8])
// BE: store <8 x i16> %[[REVE]], <8 x i16>* %{{[0-9a-zA-Z_.]+}}, align 16
// CHECK: call <8 x i16> @vec_sel(unsigned short vector[8], unsigned short vector[8], unsigned short vector[8])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_blendv_epi8(<2 x i64> noundef %{{[0-9a-zA-Z_.]+}}, <2 x i64> noundef %{{[0-9a-zA-Z_.]+}}, <2 x i64> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: call <16 x i8> @vec_splats(unsigned char)(i8 noundef zeroext 7)
// CHECK: call <16 x i8> @vec_sra(unsigned char vector[16], unsigned char vector[16])
// CHECK: call <16 x i8> @vec_sel(unsigned char vector[16], unsigned char vector[16], unsigned char vector[16])

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

// To test smmintrin.h includes tmmintrin.h

void __attribute__((noinline))
test_abs_ssse3() {
  _mm_abs_epi16(m1);
}

// CHECK-LABEL: @test_abs_ssse3

// CHECK-LABEL: define available_externally <2 x i64> @_mm_abs_epi16(<2 x i64> noundef %{{[0-9a-zA-Z_.]+}})
