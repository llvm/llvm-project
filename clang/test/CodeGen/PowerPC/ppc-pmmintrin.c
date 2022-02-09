// REQUIRES: powerpc-registered-target

// RUN: %clang -S -emit-llvm -target powerpc64-gnu-linux -mcpu=pwr8 -DNO_MM_MALLOC -ffreestanding -DNO_WARN_X86_INTRINSICS %s \
// RUN:   -fno-discard-value-names -mllvm -disable-llvm-optzns -o - | llvm-cxxfilt -n | FileCheck %s
// RUN: %clang -S -emit-llvm -target powerpc64le-gnu-linux -mcpu=pwr8 -DNO_MM_MALLOC -ffreestanding -DNO_WARN_X86_INTRINSICS %s \
// RUN:   -fno-discard-value-names -mllvm -disable-llvm-optzns -o - | llvm-cxxfilt -n | FileCheck %s

// RUN: %clang -S -emit-llvm -target powerpc64-unknown-freebsd13.0 -mcpu=pwr8 -DNO_MM_MALLOC -ffreestanding -DNO_WARN_X86_INTRINSICS %s \
// RUN:   -fno-discard-value-names -mllvm -disable-llvm-optzns -o - | llvm-cxxfilt -n | FileCheck %s
// RUN: %clang -S -emit-llvm -target powerpc64le-unknown-freebsd13.0 -mcpu=pwr8 -DNO_MM_MALLOC -ffreestanding -DNO_WARN_X86_INTRINSICS %s \
// RUN:   -fno-discard-value-names -mllvm -disable-llvm-optzns -o - | llvm-cxxfilt -n | FileCheck %s

#include <pmmintrin.h>

__m128d resd, md1, md2;
__m128 res, m1, m2;
__m128i resi, mi;
double *d;

void __attribute__((noinline))
test_pmmintrin() {
  resd = _mm_addsub_pd(md1, md2);
  res = _mm_addsub_ps(m1, m2);
  resd = _mm_hadd_pd(md1, md2);
  res = _mm_hadd_ps(m1, m2);
  resd = _mm_hsub_pd(md1, md2);
  res = _mm_hsub_ps(m1, m2);
  resi = _mm_lddqu_si128(&mi);
  resd = _mm_loaddup_pd(d);
  resd = _mm_movedup_pd(md1);
  res = _mm_movehdup_ps(m1);
  res = _mm_moveldup_ps(m1);
}

// CHECK-LABEL: define available_externally <2 x double> @_mm_addsub_pd(<2 x double> noundef %{{[0-9a-zA-Z_.]+}}, <2 x double> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: call <2 x double> @vec_xor(double vector[2], double vector[2])(<2 x double> noundef %{{[0-9a-zA-Z_.]+}}, <2 x double> noundef <double -0.000000e+00, double 0.000000e+00>)
// CHECK: call <2 x double> @vec_add(double vector[2], double vector[2])

// CHECK-LABEL: define available_externally <4 x float> @_mm_addsub_ps(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, <4 x float> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: call <4 x float> @vec_xor(float vector[4], float vector[4])(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, <4 x float> noundef <float -0.000000e+00, float 0.000000e+00, float -0.000000e+00, float 0.000000e+00>)
// CHECK: call <4 x float> @vec_add(float vector[4], float vector[4])

// CHECK-LABEL: define available_externally <2 x double> @_mm_hadd_pd(<2 x double> noundef %{{[0-9a-zA-Z_.]+}}, <2 x double> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: %[[CALL:[0-9a-zA-Z_.]+]] = call <2 x double> @vec_mergeh(double vector[2], double vector[2])
// CHECK: %[[CALL1:[0-9a-zA-Z_.]+]] = call <2 x double> @vec_mergel(double vector[2], double vector[2])
// CHECK: %[[CALL2:[0-9a-zA-Z_.]+]] = call <2 x double> @vec_add(double vector[2], double vector[2])(<2 x double> noundef %[[CALL]], <2 x double> noundef %[[CALL1]])

// CHECK-LABEL: define available_externally <4 x float> @_mm_hadd_ps(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, <4 x float> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: store <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 8, i8 9, i8 10, i8 11, i8 16, i8 17, i8 18, i8 19, i8 24, i8 25, i8 26, i8 27>, <16 x i8>* %{{[0-9a-zA-Z_.]+}}, align 16
// CHECK: store <16 x i8> <i8 4, i8 5, i8 6, i8 7, i8 12, i8 13, i8 14, i8 15, i8 20, i8 21, i8 22, i8 23, i8 28, i8 29, i8 30, i8 31>, <16 x i8>* %{{[0-9a-zA-Z_.]+}}, align 16
// CHECK: %[[CALL:[0-9a-zA-Z_.]+]] = call <4 x float> @vec_perm(float vector[4], float vector[4], unsigned char vector[16])
// CHECK: %[[CALL1:[0-9a-zA-Z_.]+]] = call <4 x float> @vec_perm(float vector[4], float vector[4], unsigned char vector[16])
// CHECK: %[[CALL2:[0-9a-zA-Z_.]+]] = call <4 x float> @vec_add(float vector[4], float vector[4])(<4 x float> noundef %[[CALL]], <4 x float> noundef %[[CALL1]])

// CHECK-LABEL: define available_externally <2 x double> @_mm_hsub_pd(<2 x double> noundef %{{[0-9a-zA-Z_.]+}}, <2 x double> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: %[[CALL:[0-9a-zA-Z_.]+]] = call <2 x double> @vec_mergeh(double vector[2], double vector[2])
// CHECK: %[[CALL1:[0-9a-zA-Z_.]+]] = call <2 x double> @vec_mergel(double vector[2], double vector[2])
// CHECK: %[[CALL2:[0-9a-zA-Z_.]+]] = call <2 x double> @vec_sub(double vector[2], double vector[2])(<2 x double> noundef %[[CALL]], <2 x double> noundef %[[CALL1]])

// CHECK-LABEL: define available_externally <4 x float> @_mm_hsub_ps(<4 x float> noundef %{{[0-9a-zA-Z_.]+}}, <4 x float> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: store <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 8, i8 9, i8 10, i8 11, i8 16, i8 17, i8 18, i8 19, i8 24, i8 25, i8 26, i8 27>, <16 x i8>* %{{[0-9a-zA-Z_.]+}}, align 16
// CHECK: store <16 x i8> <i8 4, i8 5, i8 6, i8 7, i8 12, i8 13, i8 14, i8 15, i8 20, i8 21, i8 22, i8 23, i8 28, i8 29, i8 30, i8 31>, <16 x i8>* %{{[0-9a-zA-Z_.]+}}, align 16
// CHECK: %[[CALL:[0-9a-zA-Z_.]+]] = call <4 x float> @vec_perm(float vector[4], float vector[4], unsigned char vector[16])
// CHECK: %[[CALL1:[0-9a-zA-Z_.]+]] = call <4 x float> @vec_perm(float vector[4], float vector[4], unsigned char vector[16])
// CHECK: %[[CALL2:[0-9a-zA-Z_.]+]] = call <4 x float> @vec_sub(float vector[4], float vector[4])(<4 x float> noundef %[[CALL]], <4 x float> noundef %[[CALL1]])

// CHECK-LABEL: define available_externally <2 x i64> @_mm_lddqu_si128(<2 x i64>* noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: call <4 x i32> @vec_vsx_ld(int, int const*)(i32 noundef signext 0, i32* noundef %{{[0-9a-zA-Z_.]+}})

// CHECK-LABEL: define available_externally <2 x double> @_mm_loaddup_pd(double* noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: call <2 x double> @vec_splats(double)

// CHECK-LABEL: define available_externally <2 x double> @_mm_movedup_pd(<2 x double> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: call <2 x double> @_mm_shuffle_pd(<2 x double> noundef %{{[0-9a-zA-Z_.]+}}, <2 x double> noundef %{{[0-9a-zA-Z_.]+}}, i32 noundef signext 0)

// CHECK-LABEL: define available_externally <4 x float> @_mm_movehdup_ps(<4 x float> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: call <4 x i32> @vec_mergeo(unsigned int vector[4], unsigned int vector[4])

// CHECK-LABEL: define available_externally <4 x float> @_mm_moveldup_ps(<4 x float> noundef %{{[0-9a-zA-Z_.]+}})
// CHECK: call <4 x i32> @vec_mergee(unsigned int vector[4], unsigned int vector[4])
