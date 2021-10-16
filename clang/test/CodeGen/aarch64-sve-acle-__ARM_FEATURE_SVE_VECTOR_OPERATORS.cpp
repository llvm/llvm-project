// RUN: %clang_cc1 -x c++ -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s -msve-vector-bits=128  | FileCheck %s -D#VBITS=128  --check-prefixes=CHECK,CHECK128
// RUN: %clang_cc1 -x c++ -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s -msve-vector-bits=256  | FileCheck %s -D#VBITS=256  --check-prefixes=CHECK,CHECKWIDE
// RUN: %clang_cc1 -x c++ -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s -msve-vector-bits=512  | FileCheck %s -D#VBITS=512  --check-prefixes=CHECK,CHECKWIDE
// RUN: %clang_cc1 -x c++ -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s -msve-vector-bits=1024 | FileCheck %s -D#VBITS=1024 --check-prefixes=CHECK,CHECKWIDE
// RUN: %clang_cc1 -x c++ -triple aarch64-none-linux-gnu -target-feature +sve -fallow-half-arguments-and-returns -S -O1 -Werror -Wall -emit-llvm -o - %s -msve-vector-bits=2048 | FileCheck %s -D#VBITS=2048 --check-prefixes=CHECK,CHECKWIDE
// REQUIRES: aarch64-registered-target

// Examples taken from section "3.7.3.3 Behavior specific to SVE
// vectors" of the SVE ACLE (Version 00bet6) that can be found at
// https://developer.arm.com/documentation/100987/latest
//
// Example has been expanded to work with mutiple values of
// -msve-vector-bits.

#include <arm_sve.h>

// Page 26, first paragraph of 3.7.3.3: sizeof and alignof
#if __ARM_FEATURE_SVE_BITS
#define N __ARM_FEATURE_SVE_BITS
typedef svfloat32_t fixed_svfloat __attribute__((arm_sve_vector_bits(N)));
void test01() {
  static_assert(alignof(fixed_svfloat) == 16,
                "Invalid align of Vector Length Specific Type.");
  static_assert(sizeof(fixed_svfloat) == N / 8,
                "Invalid size of Vector Length Specific Type.");
}
#endif

// Page 26, items 1 and 2 of 3.7.3.3: how VLST and GNUT are related.
#if __ARM_FEATURE_SVE_BITS && __ARM_FEATURE_SVE_VECTOR_OPERATORS
#define N __ARM_FEATURE_SVE_BITS
typedef svfloat64_t fixed_svfloat64 __attribute__((arm_sve_vector_bits(N)));
typedef float64_t gnufloat64 __attribute__((vector_size(N / 8)));
void test02() {
  static_assert(alignof(fixed_svfloat64) == alignof(gnufloat64),
                "Align of Vector Length Specific Type and GNU Vector Types "
                "should be the same.");
  static_assert(sizeof(fixed_svfloat64) == sizeof(gnufloat64),
                "Size of Vector Length Specific Type and GNU Vector Types "
                "should be the same.");
}
#endif

// Page 27, item 1.
#if __ARM_FEATURE_SVE_BITS && __ARM_FEATURE_SVE_VECTOR_OPERATORS
#define N __ARM_FEATURE_SVE_BITS
// CHECK-LABEL: define{{.*}} <vscale x 4 x i32> @_Z1f9__SVE_VLSIu11__SVInt32_tLj
// CHECK-SAME:    [[#VBITS]]
// CHECK-SAME:    EES_(<vscale x 4 x i32> noundef %x.coerce, <vscale x 4 x i32> noundef %y.coerce)
// CHECK-NEXT: entry:
// CHECK-NEXT:   [[X:%.*]] = call <[[#div(VBITS, 32)]] x i32> @llvm.experimental.vector.extract.v[[#div(VBITS, 32)]]i32.nxv4i32(<vscale x 4 x i32> [[X_COERCE:%.*]], i64 0)
// CHECK-NEXT:   [[Y:%.*]] = call <[[#div(VBITS, 32)]] x i32> @llvm.experimental.vector.extract.v[[#div(VBITS, 32)]]i32.nxv4i32(<vscale x 4 x i32> [[X_COERCE1:%.*]], i64 0)
// CHECK-NEXT:   [[ADD:%.*]] = add <[[#div(VBITS, 32)]] x i32> [[Y]], [[X]]
// CHECK-NEXT:   [[CASTSCALABLESVE:%.*]] = call <vscale x 4 x i32> @llvm.experimental.vector.insert.nxv4i32.v[[#div(VBITS, 32)]]i32(<vscale x 4 x i32> undef, <[[#div(VBITS, 32)]] x i32> [[ADD]], i64 0)
// CHECK-NEXT:   ret <vscale x 4 x i32> [[CASTSCALABLESVE]]
typedef svint32_t vec __attribute__((arm_sve_vector_bits(N)));
auto f(vec x, vec y) { return x + y; } // Returns a vec.
#endif

// Page 27, item 3, adapted for a generic value of __ARM_FEATURE_SVE_BITS
#if __ARM_FEATURE_SVE_BITS && __ARM_FEATURE_SVE_VECTOR_OPERATORS
#define N __ARM_FEATURE_SVE_BITS
typedef int16_t vec1 __attribute__((vector_size(N / 8)));
void f(vec1);
typedef svint16_t vec2 __attribute__((arm_sve_vector_bits(N)));
// CHECK-LABEL: define{{.*}} void @_Z1g9__SVE_VLSIu11__SVInt16_tLj
// CHECK-SAME:    [[#VBITS]]
// CHECK-SAME:    EE(<vscale x 8 x i16> noundef %x.coerce)
// CHECK-NEXT: entry:
// CHECK128-NEXT:   [[X:%.*]] = call <8 x i16> @llvm.experimental.vector.extract.v8i16.nxv8i16(<vscale x 8 x i16> [[X_COERCE:%.*]], i64 0)
// CHECK128-NEXT:   call void @_Z1fDv8_s(<8 x i16> noundef [[X]]) [[ATTR5:#.*]]
// CHECK128-NEXT:   ret void
// CHECKWIDE-NEXT:   [[INDIRECT_ARG_TEMP:%.*]] = alloca <[[#div(VBITS, 16)]] x i16>, align 16
// CHECKWIDE-NEXT:   [[X:%.*]] = call <[[#div(VBITS, 16)]] x i16> @llvm.experimental.vector.extract.v[[#div(VBITS, 16)]]i16.nxv8i16(<vscale x 8 x i16> [[X_COERCE:%.*]], i64 0)
// CHECKWIDE-NEXT:   store <[[#div(VBITS, 16)]] x i16> [[X]], <[[#div(VBITS, 16)]] x i16>* [[INDIRECT_ARG_TEMP]], align 16, [[TBAA6:!tbaa !.*]]
// CHECKWIDE-NEXT:   call void @_Z1fDv[[#div(VBITS, 16)]]_s(<[[#div(VBITS, 16)]] x i16>* noundef nonnull [[INDIRECT_ARG_TEMP]]) [[ATTR5:#.*]]
// CHECKWIDE-NEXT:   ret void
void g(vec2 x) { f(x); } // OK
#endif
