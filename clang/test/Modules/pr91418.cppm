// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 -x c++-header %t/foo.h \
// RUN:     -emit-pch -o %t/foo.pch
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 %t/use.cpp -include-pch \
// RUN:     %t/foo.pch -emit-llvm -o - | FileCheck %t/use.cpp

//--- foo.h
#ifndef FOO_H
#define FOO_H
typedef float __m128 __attribute__((__vector_size__(16), __aligned__(16)));

static __inline__ __m128 __attribute__((__always_inline__, __min_vector_width__(128)))
_mm_setr_ps(float __z, float __y, float __x, float __w)
{
  return __extension__ (__m128){ __z, __y, __x, __w };
}

typedef __m128 VR;

inline VR MakeVR( float X, float Y, float Z, float W )
{
 return _mm_setr_ps( X, Y, Z, W );
}

extern "C" float sqrtf(float);

namespace VectorSinConstantsSSE
{
  float a = (16 * sqrtf(0.225f));
  VR A = MakeVR(a, a, a, a);
  static const float b = (16 * sqrtf(0.225f));
  static const VR B = MakeVR(b, b, b, b);
}

#endif // FOO_H

//--- use.cpp
#include "foo.h"
float use() {
    return VectorSinConstantsSSE::A[0] + VectorSinConstantsSSE::A[1] +
           VectorSinConstantsSSE::A[2] + VectorSinConstantsSSE::A[3] +
           VectorSinConstantsSSE::B[0] + VectorSinConstantsSSE::B[1] +
           VectorSinConstantsSSE::B[2] + VectorSinConstantsSSE::B[3];
}

// CHECK: define{{.*}}@__cxx_global_var_init(
// CHECK: store{{.*}}, ptr @_ZN21VectorSinConstantsSSE1aE

// CHECK: define{{.*}}@__cxx_global_var_init.1(
// CHECK: store{{.*}}, ptr @_ZN21VectorSinConstantsSSE1AE

// CHECK: define{{.*}}@__cxx_global_var_init.2(
// CHECK: store{{.*}}, ptr @_ZN21VectorSinConstantsSSEL1BE

// CHECK: define{{.*}}@__cxx_global_var_init.3(
// CHECK: store{{.*}}, ptr @_ZN21VectorSinConstantsSSEL1bE

// CHECK: @_GLOBAL__sub_I_use.cpp
// CHECK: call{{.*}}@__cxx_global_var_init(
// CHECK: call{{.*}}@__cxx_global_var_init.1(
// CHECK: call{{.*}}@__cxx_global_var_init.3(
// CHECK: call{{.*}}@__cxx_global_var_init.2(
