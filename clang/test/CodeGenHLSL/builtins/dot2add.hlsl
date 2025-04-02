// RUN: %clang_cc1 -finclude-default-header -fnative-half-type -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,CHECK-DXIL
// RUN: %clang_cc1 -finclude-default-header -fnative-half-type -triple \
// RUN:   spirv-pc-vulkan-compute %s -emit-llvm -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,CHECK-SPIRV

// Test basic lowering to runtime function call.

// CHECK-LABEL: define {{.*}}test_default_parameter_type
float test_default_parameter_type(half2 p1, half2 p2, float p3) {
  // CHECK-SPIRV:  %[[MUL:.*]] = call reassoc nnan ninf nsz arcp afn half @llvm.spv.fdot.v2f16(<2 x half> %7, <2 x half> %8)
  // CHECK-SPIRV:  %[[CONV:.*]] = fpext reassoc nnan ninf nsz arcp afn half %[[MUL]] to float
  // CHECK-SPIRV:  %[[C:.*]] = load float, ptr %c.addr.i, align 4
  // CHECK-SPIRV:  %[[RES:.*]] = fadd reassoc nnan ninf nsz arcp afn float %[[CONV]], %[[C]]
  // CHECK-DXIL:  %[[RES:.*]] = call {{.*}} float @llvm.dx.dot2add.v2f16(<2 x half> %6, <2 x half> %7, float %8)
  // CHECK:  ret float %[[RES]]
  return dot2add(p1, p2, p3);
}

// CHECK-LABEL: define {{.*}}test_float_arg2_type
float test_float_arg2_type(half2 p1, float2 p2, float p3) {
  // CHECK-SPIRV:  %conv = fptrunc reassoc nnan ninf nsz arcp afn <2 x float> %2 to <2 x half> 
  // CHECK-SPIRV:  %[[MUL:.*]] = call reassoc nnan ninf nsz arcp afn half @llvm.spv.fdot.v2f16(<2 x half> %7, <2 x half> %8)
  // CHECK-SPIRV:  %[[CONV:.*]] = fpext reassoc nnan ninf nsz arcp afn half %[[MUL]] to float
  // CHECK-SPIRV:  %[[C:.*]] = load float, ptr %c.addr.i, align 4
  // CHECK-SPIRV:  %[[RES:.*]] = fadd reassoc nnan ninf nsz arcp afn float %[[CONV]], %[[C]]
  // CHECK-DXIL:  %conv = fptrunc reassoc nnan ninf nsz arcp afn <2 x float> %1 to <2 x half>
  // CHECK-DXIL:  %[[RES:.*]] = call {{.*}} float @llvm.dx.dot2add.v2f16(<2 x half> %6, <2 x half> %7, float %8)
  // CHECK:  ret float %[[RES]]
  return dot2add(p1, p2, p3);
}

// CHECK-LABEL: define {{.*}}test_float_arg1_type
float test_float_arg1_type(float2 p1, half2 p2, float p3) {
  // CHECK-SPIRV:  %conv = fptrunc reassoc nnan ninf nsz arcp afn <2 x float> %1 to <2 x half>
  // CHECK-SPIRV:  %[[MUL:.*]] = call reassoc nnan ninf nsz arcp afn half @llvm.spv.fdot.v2f16(<2 x half> %7, <2 x half> %8)
  // CHECK-SPIRV:  %[[CONV:.*]] = fpext reassoc nnan ninf nsz arcp afn half %[[MUL]] to float
  // CHECK-SPIRV:  %[[C:.*]] = load float, ptr %c.addr.i, align 4
  // CHECK-SPIRV:  %[[RES:.*]] = fadd reassoc nnan ninf nsz arcp afn float %[[CONV]], %[[C]]
  // CHECK-DXIL:  %conv = fptrunc reassoc nnan ninf nsz arcp afn <2 x float> %0 to <2 x half>
  // CHECK-DXIL:  %[[RES:.*]] = call {{.*}} float @llvm.dx.dot2add.v2f16(<2 x half> %6, <2 x half> %7, float %8)
  // CHECK:  ret float %[[RES]]
  return dot2add(p1, p2, p3);
}

// CHECK-LABEL: define {{.*}}test_double_arg3_type
float test_double_arg3_type(half2 p1, half2 p2, double p3) {
  // CHECK-SPIRV:  %conv = fptrunc reassoc nnan ninf nsz arcp afn double %3 to float
  // CHECK-SPIRV:  %[[MUL:.*]] = call reassoc nnan ninf nsz arcp afn half @llvm.spv.fdot.v2f16(<2 x half> %7, <2 x half> %8)
  // CHECK-SPIRV:  %[[CONV:.*]] = fpext reassoc nnan ninf nsz arcp afn half %[[MUL]] to float
  // CHECK-SPIRV:  %[[C:.*]] = load float, ptr %c.addr.i, align 4
  // CHECK-SPIRV:  %[[RES:.*]] = fadd reassoc nnan ninf nsz arcp afn float %[[CONV]], %[[C]]
  // CHECK-DXIL:  %conv = fptrunc reassoc nnan ninf nsz arcp afn double %2 to float
  // CHECK-DXIL:  %[[RES:.*]] = call {{.*}} float @llvm.dx.dot2add.v2f16(<2 x half> %6, <2 x half> %7, float %8)
  // CHECK:  ret float %[[RES]]
  return dot2add(p1, p2, p3);
}

// CHECK-LABEL: define {{.*}}test_float_arg1_arg2_type
float test_float_arg1_arg2_type(float2 p1, float2 p2, float p3) {
  // CHECK-SPIRV:  %conv = fptrunc reassoc nnan ninf nsz arcp afn <2 x float> %1 to <2 x half>
  // CHECK-SPIRV:  %conv1 = fptrunc reassoc nnan ninf nsz arcp afn <2 x float> %2 to <2 x half>
  // CHECK-SPIRV:  %[[MUL:.*]] = call reassoc nnan ninf nsz arcp afn half @llvm.spv.fdot.v2f16(<2 x half> %7, <2 x half> %8)
  // CHECK-SPIRV:  %[[CONV:.*]] = fpext reassoc nnan ninf nsz arcp afn half %[[MUL]] to float
  // CHECK-SPIRV:  %[[C:.*]] = load float, ptr %c.addr.i, align 4
  // CHECK-SPIRV:  %[[RES:.*]] = fadd reassoc nnan ninf nsz arcp afn float %[[CONV]], %[[C]]
  // CHECK-DXIL:  %conv = fptrunc reassoc nnan ninf nsz arcp afn <2 x float> %0 to <2 x half>
  // CHECK-DXIL:  %conv1 = fptrunc reassoc nnan ninf nsz arcp afn <2 x float> %1 to <2 x half>
  // CHECK-DXIL:  %[[RES:.*]] = call {{.*}} float @llvm.dx.dot2add.v2f16(<2 x half> %6, <2 x half> %7, float %8)
  // CHECK:  ret float %[[RES]]
  return dot2add(p1, p2, p3);
}

// CHECK-LABEL: define {{.*}}test_double_arg1_arg2_type
float test_double_arg1_arg2_type(double2 p1, double2 p2, float p3) {
  // CHECK-SPIRV:  %conv = fptrunc reassoc nnan ninf nsz arcp afn <2 x double> %1 to <2 x half>
  // CHECK-SPIRV:  %conv1 = fptrunc reassoc nnan ninf nsz arcp afn <2 x double> %2 to <2 x half>
  // CHECK-SPIRV:  %[[MUL:.*]] = call reassoc nnan ninf nsz arcp afn half @llvm.spv.fdot.v2f16(<2 x half> %7, <2 x half> %8)
  // CHECK-SPIRV:  %[[CONV:.*]] = fpext reassoc nnan ninf nsz arcp afn half %[[MUL]] to float
  // CHECK-SPIRV:  %[[C:.*]] = load float, ptr %c.addr.i, align 4
  // CHECK-SPIRV:  %[[RES:.*]] = fadd reassoc nnan ninf nsz arcp afn float %[[CONV]], %[[C]]
  // CHECK-DXIL:  %conv = fptrunc reassoc nnan ninf nsz arcp afn <2 x double> %0 to <2 x half>
  // CHECK-DXIL:  %conv1 = fptrunc reassoc nnan ninf nsz arcp afn <2 x double> %1 to <2 x half>
  // CHECK-DXIL:  %[[RES:.*]] = call {{.*}} float @llvm.dx.dot2add.v2f16(<2 x half> %6, <2 x half> %7, float %8)
  // CHECK:  ret float %[[RES]]
  return dot2add(p1, p2, p3);
}

// CHECK-LABEL: define {{.*}}test_int16_arg1_arg2_type
float test_int16_arg1_arg2_type(int16_t2 p1, int16_t2 p2, float p3) {
  // CHECK-SPIRV:  %conv = sitofp <2 x i16> %1 to <2 x half>
  // CHECK-SPIRV:  %conv1 = sitofp <2 x i16> %2 to <2 x half>
  // CHECK-SPIRV:  %[[MUL:.*]] = call reassoc nnan ninf nsz arcp afn half @llvm.spv.fdot.v2f16(<2 x half> %7, <2 x half> %8)
  // CHECK-SPIRV:  %[[CONV:.*]] = fpext reassoc nnan ninf nsz arcp afn half %[[MUL]] to float
  // CHECK-SPIRV:  %[[C:.*]] = load float, ptr %c.addr.i, align 4
  // CHECK-SPIRV:  %[[RES:.*]] = fadd reassoc nnan ninf nsz arcp afn float %[[CONV]], %[[C]]
  // CHECK-DXIL:  %conv = sitofp <2 x i16> %0 to <2 x half>
  // CHECK-DXIL:  %conv1 = sitofp <2 x i16> %1 to <2 x half>
  // CHECK-DXIL:  %[[RES:.*]] = call {{.*}} float @llvm.dx.dot2add.v2f16(<2 x half> %6, <2 x half> %7, float %8)
  // CHECK:  ret float %[[RES]]
  return dot2add(p1, p2, p3);
}

// CHECK-LABEL: define {{.*}}test_int32_arg1_arg2_type
float test_int32_arg1_arg2_type(int32_t2 p1, int32_t2 p2, float p3) {
  // CHECK-SPIRV:  %conv = sitofp <2 x i32> %1 to <2 x half>
  // CHECK-SPIRV:  %conv1 = sitofp <2 x i32> %2 to <2 x half>
  // CHECK-SPIRV:  %[[MUL:.*]] = call reassoc nnan ninf nsz arcp afn half @llvm.spv.fdot.v2f16(<2 x half> %7, <2 x half> %8)
  // CHECK-SPIRV:  %[[CONV:.*]] = fpext reassoc nnan ninf nsz arcp afn half %[[MUL]] to float
  // CHECK-SPIRV:  %[[C:.*]] = load float, ptr %c.addr.i, align 4
  // CHECK-SPIRV:  %[[RES:.*]] = fadd reassoc nnan ninf nsz arcp afn float %[[CONV]], %[[C]]
  // CHECK-DXIL:  %conv = sitofp <2 x i32> %0 to <2 x half>
  // CHECK-DXIL:  %conv1 = sitofp <2 x i32> %1 to <2 x half>
  // CHECK-DXIL:  %[[RES:.*]] = call {{.*}} float @llvm.dx.dot2add.v2f16(<2 x half> %6, <2 x half> %7, float %8)
  // CHECK:  ret float %[[RES]]
  return dot2add(p1, p2, p3);
}

// CHECK-LABEL: define {{.*}}test_int64_arg1_arg2_type
float test_int64_arg1_arg2_type(int64_t2 p1, int64_t2 p2, float p3) {
  // CHECK-SPIRV:  %conv = sitofp <2 x i64> %1 to <2 x half>
  // CHECK-SPIRV:  %conv1 = sitofp <2 x i64> %2 to <2 x half>
  // CHECK-SPIRV:  %[[MUL:.*]] = call reassoc nnan ninf nsz arcp afn half @llvm.spv.fdot.v2f16(<2 x half> %7, <2 x half> %8)
  // CHECK-SPIRV:  %[[CONV:.*]] = fpext reassoc nnan ninf nsz arcp afn half %[[MUL]] to float
  // CHECK-SPIRV:  %[[C:.*]] = load float, ptr %c.addr.i, align 4
  // CHECK-SPIRV:  %[[RES:.*]] = fadd reassoc nnan ninf nsz arcp afn float %[[CONV]], %[[C]]
  // CHECK-DXIL:  %conv = sitofp <2 x i64> %0 to <2 x half>
  // CHECK-DXIL:  %conv1 = sitofp <2 x i64> %1 to <2 x half>
  // CHECK-DXIL:  %[[RES:.*]] = call {{.*}} float @llvm.dx.dot2add.v2f16(<2 x half> %6, <2 x half> %7, float %8)
  // CHECK:  ret float %[[RES]]
  return dot2add(p1, p2, p3);
}

// CHECK-LABEL: define {{.*}}test_bool_arg1_arg2_type
float test_bool_arg1_arg2_type(bool2 p1, bool2 p2, float p3) {
  // CHECK-SPIRV:  %loadedv = trunc <2 x i32> %3 to <2 x i1>
  // CHECK-SPIRV:  %conv = uitofp <2 x i1> %loadedv to <2 x half>
  // CHECK-SPIRV:  %loadedv1 = trunc <2 x i32> %4 to <2 x i1>
  // CHECK-SPIRV:  %conv2 = uitofp <2 x i1> %loadedv1 to <2 x half>
  // CHECK-SPIRV:  %[[MUL:.*]] = call reassoc nnan ninf nsz arcp afn half @llvm.spv.fdot.v2f16(<2 x half> %9, <2 x half> %10)
  // CHECK-SPIRV:  %[[CONV:.*]] = fpext reassoc nnan ninf nsz arcp afn half %[[MUL]] to float
  // CHECK-SPIRV:  %[[C:.*]] = load float, ptr %c.addr.i, align 4
  // CHECK-SPIRV:  %[[RES:.*]] = fadd reassoc nnan ninf nsz arcp afn float %[[CONV]], %[[C]]
  // CHECK-DXIL:  %loadedv = trunc <2 x i32> %2 to <2 x i1>
  // CHECK-DXIL:  %conv = uitofp <2 x i1> %loadedv to <2 x half>
  // CHECK-DXIL:  %loadedv1 = trunc <2 x i32> %3 to <2 x i1>
  // CHECK-DXIL:  %conv2 = uitofp <2 x i1> %loadedv1 to <2 x half>
  // CHECK-DXIL:  %[[RES:.*]] = call {{.*}} float @llvm.dx.dot2add.v2f16(<2 x half> %8, <2 x half> %9, float %10)
  // CHECK:  ret float %[[RES]]
  return dot2add(p1, p2, p3);
}
