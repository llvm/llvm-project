// RUN: %clang_cc1 -triple thumbv7k-apple-watchos2.0 -target-abi aapcs16 -target-cpu cortex-a7 %s -o - -emit-llvm | FileCheck %s

// REQUIRES: aarch64-registered-target || arm-registered-target

#include <arm_neon.h>

// Make sure 64 and 128 bit types are naturally aligned by the v7k ABI:

// CHECK: target datalayout = "e-m:o-p:32:32-Fi8-i64:64-a:0:32-n32-S128"

typedef struct {
  float arr[4];
} HFA;

// CHECK: define{{.*}} void @simple_hfa([4 x float] %h.coerce)
void simple_hfa(HFA h) {}

// CHECK: define{{.*}} %struct.HFA @return_simple_hfa
HFA return_simple_hfa() {}

typedef struct {
  double arr[4];
} BigHFA;

// We don't want any padding type to be included by Clang when using the
// APCS-VFP ABI, that needs to be handled by LLVM if needed.

// CHECK: void @no_padding(i32 noundef %r0, i32 noundef %r1, i32 noundef %r2, [4 x double] %d0_d3.coerce, [4 x double] %d4_d7.coerce, [4 x double] %sp.coerce, i64 noundef %split)
void no_padding(int r0, int r1, int r2, BigHFA d0_d3, BigHFA d4_d7, BigHFA sp,
                long long split) {}

// Structs larger than 16 bytes should be passed indirectly in space allocated
// by the caller (a pointer to this storage should be what occurs in the arg
// list).

typedef struct {
  float x;
  long long y;
  double z;
} BigStruct;

// CHECK: define{{.*}} void @big_struct_indirect(ptr noundef %b)
void big_struct_indirect(BigStruct b) {}

// CHECK: define{{.*}} void @return_big_struct_indirect(ptr noalias sret
BigStruct return_big_struct_indirect() {}

// Structs smaller than 16 bytes should be passed directly, and coerced to
// either [N x i32] or [N x i64] depending on alignment requirements.

typedef struct {
  float x;
  int y;
  double z;
} SmallStruct;

// CHECK: define{{.*}} void @small_struct_direct([2 x i64] %s.coerce)
void small_struct_direct(SmallStruct s) {}

// CHECK: define{{.*}} [4 x i32] @return_small_struct_direct()
SmallStruct return_small_struct_direct() {}

typedef struct {
  float x;
  int y;
  int z;
} SmallStructSmallAlign;

// CHECK: define{{.*}} void @small_struct_align_direct([3 x i32] %s.coerce)
void small_struct_align_direct(SmallStructSmallAlign s) {}

typedef struct {
  char x;
  short y;
} PaddedSmallStruct;

// CHECK: define{{.*}} i32 @return_padded_small_struct()
PaddedSmallStruct return_padded_small_struct() {}

typedef struct {
  char arr[7];
} OddlySizedStruct;

// CHECK: define{{.*}} [2 x i32] @return_oddly_sized_struct()
OddlySizedStruct return_oddly_sized_struct() {}

// CHECK: define{{.*}} <4 x float> @test_va_arg_vec(ptr noundef %l)


// CHECK: [[GEP_ALIGN:%.+]] = getelementptr inbounds i8, ptr {{%.*}}, i32 15
// CHECK: [[ALIGNED:%.*]] = call ptr @llvm.ptrmask.p0.i32(ptr [[GEP_ALIGN]], i32 -16)
// CHECK:   load <4 x float>, ptr [[ALIGNED]], align 16
float32x4_t test_va_arg_vec(__builtin_va_list l) {
  return __builtin_va_arg(l, float32x4_t);
}
