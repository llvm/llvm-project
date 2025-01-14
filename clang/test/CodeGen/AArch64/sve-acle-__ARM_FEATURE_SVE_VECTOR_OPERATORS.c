// RUN: %clang_cc1 -triple aarch64 -target-feature +sve -O1 -Werror -Wall -emit-llvm -o - %s -mvscale-min=1 -mvscale-max=1 | FileCheck %s -D#VBITS=128 --check-prefixes=CHECK128
// RUN: %clang_cc1 -triple aarch64 -target-feature +sve -O1 -Werror -Wall -emit-llvm -o - %s -mvscale-min=2 -mvscale-max=2 | FileCheck %s -D#VBITS=256 --check-prefixes=CHECK,CHECK256
// RUN: %clang_cc1 -triple aarch64 -target-feature +sve -O1 -Werror -Wall -emit-llvm -o - %s -mvscale-min=4 -mvscale-max=4 | FileCheck %s -D#VBITS=512 --check-prefixes=CHECK,CHECK512
// RUN: %clang_cc1 -triple aarch64 -target-feature +sve -O1 -Werror -Wall -emit-llvm -o - %s -mvscale-min=8 -mvscale-max=8 | FileCheck %s -D#VBITS=1024 --check-prefixes=CHECK,CHECK1024
// RUN: %clang_cc1 -triple aarch64 -target-feature +sve -O1 -Werror -Wall -emit-llvm -o - %s -mvscale-min=16 -mvscale-max=16 | FileCheck %s -D#VBITS=2048 --check-prefixes=CHECK,CHECK2048

// REQUIRES: aarch64-registered-target

// Examples taken from section "3.7.3.3 Behavior specific to SVE
// vectors" of the SVE ACLE (Version 00bet6) that can be found at
// https://developer.arm.com/documentation/100987/latest
//
// Example has been expanded to work with mutiple values of
// -mvscale-{min,max}.

#include <arm_sve.h>

// Page 27, item 1
#if __ARM_FEATURE_SVE_BITS == 256 && __ARM_FEATURE_SVE_VECTOR_OPERATORS
// CHECK256-LABEL: @x256 ={{.*}} local_unnamed_addr global <4 x i64> <i64 0, i64 1, i64 2, i64 3>, align 16
typedef svint64_t vec256 __attribute__((arm_sve_vector_bits(256)));
vec256 x256 = {0, 1, 2, 3};
#endif

#if __ARM_FEATURE_SVE_BITS == 512 && __ARM_FEATURE_SVE_VECTOR_OPERATORS
// CHECK512-LABEL: @x512 ={{.*}} local_unnamed_addr global <8 x i64> <i64 0, i64 1, i64 2, i64 3, i64 3, i64 2, i64 1, i64 0>, align 16
typedef svint64_t vec512 __attribute__((arm_sve_vector_bits(512)));
vec512 x512 = {0, 1, 2, 3, 3 , 2 , 1, 0};
#endif

#if __ARM_FEATURE_SVE_BITS == 1024 && __ARM_FEATURE_SVE_VECTOR_OPERATORS
// CHECK1024-LABEL: @x1024 ={{.*}} local_unnamed_addr global <16 x i64> <i64 0, i64 1, i64 2, i64 3, i64 3, i64 2, i64 1, i64 0, i64 0, i64 1, i64 2, i64 3, i64 3, i64 2, i64 1, i64 0>, align 16
typedef svint64_t vec1024 __attribute__((arm_sve_vector_bits(1024)));
vec1024 x1024 = {0, 1, 2, 3, 3 , 2 , 1, 0, 0, 1, 2, 3, 3 , 2 , 1, 0};
#endif

#if __ARM_FEATURE_SVE_BITS == 2048 && __ARM_FEATURE_SVE_VECTOR_OPERATORS
// CHECK2048-LABEL: @x2048 ={{.*}} local_unnamed_addr global <32 x i64> <i64 0, i64 1, i64 2, i64 3, i64 3, i64 2, i64 1, i64 0, i64 0, i64 1, i64 2, i64 3, i64 3, i64 2, i64 1, i64 0, i64 0, i64 1, i64 2, i64 3, i64 3, i64 2, i64 1, i64 0, i64 0, i64 1, i64 2, i64 3, i64 3, i64 2, i64 1, i64 0>, align 16
typedef svint64_t vec2048 __attribute__((arm_sve_vector_bits(2048)));
vec2048 x2048 = {0, 1, 2, 3, 3 , 2 , 1, 0, 0, 1, 2, 3, 3 , 2 , 1, 0,
                 0, 1, 2, 3, 3 , 2 , 1, 0, 0, 1, 2, 3, 3 , 2 , 1, 0};
#endif

// Page 27, item 2. We can not change the ABI of existing vector
// types, including vec_int8.  That's why in the SVE ACLE, VLST is
// distinct from, but mostly interchangeable with, the corresponding
// GNUT. VLST is treated for ABI purposes like an SVE type but GNUT
// continues to be a normal GNU vector type, with base Armv8-A PCS
// rules.
#if __ARM_FEATURE_SVE_BITS && __ARM_FEATURE_SVE_VECTOR_OPERATORS
#define N __ARM_FEATURE_SVE_BITS
typedef int8_t vec_int8 __attribute__((vector_size(N / 8)));
// CHECK128-LABEL: define{{.*}} <16 x i8> @f2(<16 x i8> noundef %x)
// CHECK128-NEXT:  entry:
// CHECK128-NEXT:    [[TMP0:%.*]] = tail call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 31)
// CHECK128-NEXT:    [[CASTSCALABLESVE:%.*]] = tail call <vscale x 16 x i8> @llvm.vector.insert.nxv16i8.v16i8(<vscale x 16 x i8> poison, <16 x i8> [[X:%.*]], i64 0)
// CHECK128-NEXT:    [[TMP1:%.*]] = tail call <vscale x 16 x i8> @llvm.aarch64.sve.asrd.nxv16i8(<vscale x 16 x i1> [[TMP0]], <vscale x 16 x i8> [[CASTSCALABLESVE]], i32 1)
// CHECK128-NEXT:    [[CASTFIXEDSVE:%.*]] = tail call <16 x i8> @llvm.vector.extract.v16i8.nxv16i8(<vscale x 16 x i8> [[TMP1]], i64 0)
// CHECK128-NEXT:    ret <16 x i8> [[CASTFIXEDSVE]]

// CHECK-LABEL: define{{.*}} void @f2(
// CHECK-SAME:   ptr dead_on_unwind noalias nocapture writable writeonly sret(<[[#div(VBITS,8)]] x i8>) align 16 initializes((0, [[#div(VBITS,8)]])) %agg.result, ptr nocapture noundef readonly %0)
// CHECK-NEXT: entry:
// CHECK-NEXT:   [[X:%.*]] = load <[[#div(VBITS,8)]] x i8>, ptr [[TMP0:%.*]], align 16, [[TBAA6:!tbaa !.*]]
// CHECK-NEXT:   [[TMP1:%.*]] = tail call <vscale x 16 x i1> @llvm.aarch64.sve.ptrue.nxv16i1(i32 31)
// CHECK-NEXT:   [[CASTSCALABLESVE:%.*]] = tail call <vscale x 16 x i8> @llvm.vector.insert.nxv16i8.v[[#div(VBITS,8)]]i8(<vscale x 16 x i8> poison, <[[#div(VBITS,8)]] x i8> [[X]], i64 0)
// CHECK-NEXT:   [[TMP2:%.*]] = tail call <vscale x 16 x i8> @llvm.aarch64.sve.asrd.nxv16i8(<vscale x 16 x i1> [[TMP1]], <vscale x 16 x i8> [[CASTSCALABLESVE]], i32 1)
// CHECK-NEXT:   [[CASTFIXEDSVE:%.*]] = tail call <[[#div(VBITS,8)]] x i8> @llvm.vector.extract.v[[#div(VBITS,8)]]i8.nxv16i8(<vscale x 16 x i8> [[TMP2]], i64 0)
// CHECK-NEXT:   store <[[#div(VBITS,8)]] x i8> [[CASTFIXEDSVE]], ptr [[AGG_RESULT:%.*]], align 16, [[TBAA6]]
// CHECK-NEXT:   ret void
vec_int8 f2(vec_int8 x) { return svasrd_x(svptrue_b8(), x, 1); }
#endif

// Page 27, item 3.
#if __ARM_FEATURE_SVE_BITS && __ARM_FEATURE_SVE_VECTOR_OPERATORS
#define N __ARM_FEATURE_SVE_BITS
typedef int8_t vec1 __attribute__((vector_size(N / 8)));
void f3(vec1);
typedef svint8_t vec2 __attribute__((arm_sve_vector_bits(N)));

// CHECK128-LABEL: define{{.*}} void @g(<vscale x 16 x i8> noundef %x.coerce)
// CHECK128-NEXT: entry:
// CHECK128-NEXT:  [[X:%.*]] = tail call <16 x i8> @llvm.vector.extract.v16i8.nxv16i8(<vscale x 16 x i8> [[X_COERCE:%.*]], i64 0)
// CHECK128-NEXT:    call void @f3(<16 x i8> noundef [[X]]) [[ATTR5:#.*]]
// CHECK128-NEXT:    ret void

// CHECK-LABEL: define{{.*}} void @g(<vscale x 16 x i8> noundef %x.coerce)
// CHECK-NEXT: entry:
// CHECK-NEXT:   [[INDIRECT_ARG_TEMP:%.*]] = alloca <[[#div(VBITS,8)]] x i8>, align 16
// CHECK-NEXT:   [[X:%.*]] = tail call <[[#div(VBITS,8)]] x i8> @llvm.vector.extract.v[[#div(VBITS,8)]]i8.nxv16i8(<vscale x 16 x i8> [[X_COERCE:%.*]], i64 0)
// CHECK-NEXT:   call void @llvm.lifetime.start.p0(i64 [[SIZE:[0-9]+]], ptr nonnull [[INDIRECT_ARG_TEMP]]) #[[ATTR6:[0-9]+]]
// CHECK-NEXT:   store <[[#div(VBITS,8)]] x i8> [[X]], ptr [[INDIRECT_ARG_TEMP]], align 16, [[TBAA6]]
// CHECK-NEXT:   call void @f3(ptr noundef nonnull [[INDIRECT_ARG_TEMP]]) [[ATTR5:#.*]]
// CHECK-NEXT:   call void @llvm.lifetime.end.p0(i64 [[SIZE]], ptr nonnull [[INDIRECT_ARG_TEMP]]) #[[ATTR6:[0-9]+]]
// CHECK-NEXT:   ret void

// CHECK128-LABEL: declare void @f3(<16 x i8> noundef)

// CHECK-LABEL: declare void @f3(
// CHECK-SAME:   ptr noundef)
void g(vec2 x) { f3(x); } // OK
#endif
