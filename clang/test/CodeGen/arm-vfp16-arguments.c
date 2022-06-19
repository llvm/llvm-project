// RUN: %clang_cc1 -no-opaque-pointers -triple armv7a--none-eabi -target-abi aapcs \
// RUN:   -mfloat-abi soft -target-feature +neon -emit-llvm -o - -O1 %s \
// RUN:   | FileCheck %s --check-prefix=CHECK-SOFT
// RUN: %clang_cc1 -no-opaque-pointers -triple armv7a--none-eabi -target-abi aapcs \
// RUN:   -mfloat-abi hard -target-feature +neon -emit-llvm -o - -O1 %s \
// RUN:   | FileCheck %s --check-prefix=CHECK-HARD
// RUN: %clang_cc1 -no-opaque-pointers -triple armv7a--none-eabi -target-abi aapcs \
// RUN:   -mfloat-abi hard -target-feature +neon -target-feature +fullfp16 \
// RUN:   -emit-llvm -o - -O1 %s \
// RUN:   | FileCheck %s --check-prefix=CHECK-FULL

typedef __attribute__((neon_vector_type(4))) __fp16 float16x4_t;
typedef __attribute__((neon_vector_type(8))) __fp16 float16x8_t;

typedef struct { float16x4_t x[2]; } hfa_t;
// CHECK-FULL: %struct.hfa_t = type { [2 x <4 x half>] }

float16x4_t g4;
float16x8_t g8;

void st4(float16x4_t a) { g4 = a; }
// CHECK-SOFT: define{{.*}} void @st4(<2 x i32> noundef %a.coerce)
// CHECK-SOFT: store <2 x i32> %a.coerce, <2 x i32>* bitcast (<4 x half>* @g4 to <2 x i32>*)
//
// CHECK-HARD: define{{.*}} arm_aapcs_vfpcc void @st4(<2 x i32> noundef %a.coerce)
// CHECK-HARD: store <2 x i32> %a.coerce, <2 x i32>* bitcast (<4 x half>* @g4 to <2 x i32>*)
//
// CHECK-FULL: define{{.*}} arm_aapcs_vfpcc void @st4(<4 x half> noundef %a)
// CHECK-FULL: store <4 x half> %a, <4 x half>* @g4

float16x4_t ld4(void) { return g4; }
// CHECK-SOFT: define{{.*}} <2 x i32> @ld4()
// CHECK-SOFT: %0 = load <2 x i32>, <2 x i32>* bitcast (<4 x half>* @g4 to <2 x i32>*)
// CHECK-SOFT: ret <2 x i32> %0
//
// CHECK-HARD: define{{.*}} arm_aapcs_vfpcc <2 x i32> @ld4()
// CHECK-HARD: %0 = load <2 x i32>, <2 x i32>* bitcast (<4 x half>* @g4 to <2 x i32>*)
// CHECK-HARD: ret <2 x i32> %0
//
// CHECK-FULL: define{{.*}} arm_aapcs_vfpcc <4 x half> @ld4()
// CHECK-FULL: %0 = load <4 x half>, <4 x half>* @g4
// CHECK-FULL: ret <4 x half> %0

void st8(float16x8_t a) { g8 = a; }
// CHECK-SOFT: define{{.*}} void @st8(<4 x i32> noundef %a.coerce)
// CHECK-SOFT: store <4 x i32> %a.coerce, <4 x i32>* bitcast (<8 x half>* @g8 to <4 x i32>*)
//
// CHECK-HARD: define{{.*}} arm_aapcs_vfpcc void @st8(<4 x i32> noundef %a.coerce)
// CHECK-HARD: store <4 x i32> %a.coerce, <4 x i32>* bitcast (<8 x half>* @g8 to <4 x i32>*)
//
// CHECK-FULL: define{{.*}} arm_aapcs_vfpcc void @st8(<8 x half> noundef %a)
// CHECK-FULL: store <8 x half> %a, <8 x half>* @g8

float16x8_t ld8(void) { return g8; }
// CHECK-SOFT: define{{.*}} <4 x i32> @ld8()
// CHECK-SOFT: %0 = load <4 x i32>, <4 x i32>* bitcast (<8 x half>* @g8 to <4 x i32>*)
// CHECK-SOFT: ret <4 x i32> %0
//
// CHECK-HARD: define{{.*}} arm_aapcs_vfpcc <4 x i32> @ld8()
// CHECK-HARD: %0 = load <4 x i32>, <4 x i32>* bitcast (<8 x half>* @g8 to <4 x i32>*)
// CHECK-HARD: ret <4 x i32> %0
//
// CHECK-FULL: define{{.*}} arm_aapcs_vfpcc <8 x half> @ld8()
// CHECK-FULL: %0 = load <8 x half>, <8 x half>* @g8
// CHECK-FULL: ret <8 x half> %0

void test_hfa(hfa_t a) {}
// CHECK-SOFT: define{{.*}} void @test_hfa([2 x i64] %a.coerce)
// CHECK-HARD: define{{.*}} arm_aapcs_vfpcc void @test_hfa([2 x <2 x i32>] %a.coerce)
// CHECK-FULL: define{{.*}} arm_aapcs_vfpcc void @test_hfa(%struct.hfa_t %a.coerce)

hfa_t ghfa;
hfa_t test_ret_hfa(void) { return ghfa; }
// CHECK-SOFT: define{{.*}} void @test_ret_hfa(%struct.hfa_t* noalias nocapture writeonly sret(%struct.hfa_t) align 8 %agg.result)
// CHECK-HARD: define{{.*}} arm_aapcs_vfpcc [2 x <2 x i32>] @test_ret_hfa()
// CHECK-FULL: define{{.*}} arm_aapcs_vfpcc %struct.hfa_t @test_ret_hfa()
