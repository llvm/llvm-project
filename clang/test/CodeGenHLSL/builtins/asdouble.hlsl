// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,CHECK-DXIL
// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   spirv-pc-vulkan-library %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,CHECK-SPV

// Test lowering of asdouble expansion to shuffle/bitcast and splat when required

// CHECK-LABEL: define {{.*}}test_uint
// CHECK-SAME: (i32 noundef %[[LOW_ARG:[^,]+]], i32 noundef %[[HIGH_ARG:[^)]+]])
double test_uint(uint low, uint high) {
  // CHECK: store i32 %[[LOW_ARG]], ptr %[[LOW_ADDR:.*]], align 4
  // CHECK: store i32 %[[HIGH_ARG]], ptr %[[HIGH_ADDR:.*]], align 4
  // CHECK: %[[LOW:.*]] = load i32, ptr %[[LOW_ADDR]], align 4
  // CHECK: %[[HIGH:.*]] = load i32, ptr %[[HIGH_ADDR]], align 4
  // CHECK-SPV: %[[LOW_INSERT:.*]] = insertelement <1 x i32> poison, i32 %[[LOW]], i64 0
  // CHECK-SPV: %[[LOW_SHUFFLE:.*]] = shufflevector <1 x i32> %[[LOW_INSERT]], {{.*}} zeroinitializer
  // CHECK-SPV: %[[HIGH_INSERT:.*]] = insertelement <1 x i32> poison, i32 %[[HIGH]], i64 0
  // CHECK-SPV: %[[HIGH_SHUFFLE:.*]] = shufflevector <1 x i32> %[[HIGH_INSERT]], {{.*}} zeroinitializer

  // CHECK-SPV:      %[[SHUFFLE0:.*]] = shufflevector <1 x i32> %[[LOW_SHUFFLE]], <1 x i32> %[[HIGH_SHUFFLE]],
  // CHECK-SPV-SAME: {{.*}} <i32 0, i32 1>
  // CHECK-SPV:      bitcast <2 x i32> %[[SHUFFLE0]] to double

  // CHECK-DXIL: call reassoc nnan ninf nsz arcp afn double @llvm.dx.asdouble.i32(i32 %[[LOW]], i32 %[[HIGH]])
  return asdouble(low, high);
}

// CHECK-DXIL: declare double @llvm.dx.asdouble.i32

// CHECK-LABEL: define {{.*}}test_vuint
// CHECK-SAME: (<3 x i32> noundef %[[V3_LOW_ARG:[^,]+]], <3 x i32> noundef %[[V3_HIGH_ARG:[^)]+]])
double3 test_vuint(uint3 low, uint3 high) {
  // CHECK: store <3 x i32> %[[V3_LOW_ARG]], ptr %[[V3_LOW_ADDR:.*]], align 4
  // CHECK: store <3 x i32> %[[V3_HIGH_ARG]], ptr %[[V3_HIGH_ADDR:.*]], align 4
  // CHECK: %[[V3_LOW:.*]] = load <3 x i32>, ptr %[[V3_LOW_ADDR]], align 4
  // CHECK: %[[V3_HIGH:.*]] = load <3 x i32>, ptr %[[V3_HIGH_ADDR]], align 4
  // CHECK-SPV:      %[[SHUFFLE1:.*]] = shufflevector <3 x i32> %[[V3_LOW]], <3 x i32> %[[V3_HIGH]],
  // CHECK-SPV-SAME: {{.*}} <i32 0, i32 3, i32 1, i32 4, i32 2, i32 5>
  // CHECK-SPV:      bitcast <6 x i32> %[[SHUFFLE1]] to <3 x double>

  // CHECK-DXIL: call reassoc nnan ninf nsz arcp afn <3 x double> @llvm.dx.asdouble.v3i32(<3 x i32> %[[V3_LOW]], <3 x i32> %[[V3_HIGH]])
  return asdouble(low, high);
}

// CHECK-DXIL: declare <3 x double> @llvm.dx.asdouble.v3i32

// CHECK-LABEL: define {{.*}}test_vuint5
// CHECK-SAME: (<5 x i32> noundef %[[V5_LOW_ARG:[^,]+]], <5 x i32> noundef %[[V5_HIGH_ARG:[^)]+]])
vector<double, 5> test_vuint5(vector<uint, 5> low, vector<uint, 5> high) {
  // CHECK: store <5 x i32> %[[V5_LOW_ARG]], ptr %[[V5_LOW_ADDR:.*]], align 4
  // CHECK: store <5 x i32> %[[V5_HIGH_ARG]], ptr %[[V5_HIGH_ADDR:.*]], align 4
  // CHECK: %[[V5_LOW:.*]] = load <5 x i32>, ptr %[[V5_LOW_ADDR]], align 4
  // CHECK: %[[V5_HIGH:.*]] = load <5 x i32>, ptr %[[V5_HIGH_ADDR]], align 4
  // CHECK-SPV:      %[[SHUFFLE2:.*]] = shufflevector <5 x i32> %[[V5_LOW]], <5 x i32> %[[V5_HIGH]],
  // CHECK-SPV-SAME: {{.*}} <i32 0, i32 5, i32 1, i32 6, i32 2, i32 7, i32 3, i32 8, i32 4, i32 9>
  // CHECK-SPV:      bitcast <10 x i32> %[[SHUFFLE2]] to <5 x double>

  // CHECK-DXIL: call reassoc nnan ninf nsz arcp afn <5 x double> @llvm.dx.asdouble.v5i32(<5 x i32> %[[V5_LOW]], <5 x i32> %[[V5_HIGH]])
  return asdouble(low, high);
}

// CHECK-DXIL: declare <5 x double> @llvm.dx.asdouble.v5i32
