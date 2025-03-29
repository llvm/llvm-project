// RUN: %clang_cc1 -finclude-default-header -fnative-half-type -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,CHECK-DXIL
// RUN: %clang_cc1 -finclude-default-header -fnative-half-type -triple \
// RUN:   spirv-pc-vulkan-compute %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,CHECK-SPIRV

// Test basic lowering to runtime function call.

// CHECK-LABEL: define {{.*}}test
float test(half2 p1, half2 p2, float p3) {
  // CHECK-SPIRV:  %[[RES:.*]] = call {{.*}} float @_ZN4hlsl7dot2addEDv2_DhS0_f(<2 x half> {{.*}} %1, <2 x half> {{.*}} %2, float {{.*}} %3) #3 {{.*}}
  // CHECK-DXIL:  %[[RES:.*]] = call {{.*}} float @_ZN4hlsl7dot2addEDv2_DhS0_f(<2 x half> {{.*}} %0, <2 x half> {{.*}} %1, float {{.*}} %2) #2
  // CHECK:  ret float %[[RES]]
  return dot2add(p1, p2, p3);
}

// CHECK-LABEL: define {{.*}}test_float_arg2_type
float test_float_arg2_type(half2 p1, float2 p2, float p3) {
  // CHECK-SPIRV:  %conv = fptrunc reassoc nnan ninf nsz arcp afn <2 x float> %2 to <2 x half>
  // CHECK-SPIRV:  %[[RES:.*]] = call {{.*}} float @_ZN4hlsl7dot2addEDv2_DhS0_f(<2 x half> {{.*}} %1, <2 x half> {{.*}} %conv, float {{.*}} %3) #3 {{.*}}
  // CHECK-DXIL:  %conv = fptrunc reassoc nnan ninf nsz arcp afn <2 x float> %1 to <2 x half>
  // CHECK-DXIL:  %[[RES:.*]] = call {{.*}} float @_ZN4hlsl7dot2addEDv2_DhS0_f(<2 x half> {{.*}} %0, <2 x half> {{.*}} %conv, float {{.*}} %2) #2
  // CHECK:  ret float %[[RES]]
  return dot2add(p1, p2, p3);
}

// CHECK-LABEL: define {{.*}}test_float_arg1_type
float test_float_arg1_type(float2 p1, half2 p2, float p3) {
  // CHECK-SPIRV:  %conv = fptrunc reassoc nnan ninf nsz arcp afn <2 x float> %1 to <2 x half>
  // CHECK-SPIRV:  %[[RES:.*]] = call {{.*}} float @_ZN4hlsl7dot2addEDv2_DhS0_f(<2 x half> {{.*}} %conv, <2 x half> {{.*}} %2, float {{.*}} %3) #3 {{.*}}
  // CHECK-DXIL:  %conv = fptrunc reassoc nnan ninf nsz arcp afn <2 x float> %0 to <2 x half>
  // CHECK-DXIL:  %[[RES:.*]] = call {{.*}} float @_ZN4hlsl7dot2addEDv2_DhS0_f(<2 x half> {{.*}} %conv, <2 x half> {{.*}} %1, float {{.*}} %2) #2
  // CHECK:  ret float %[[RES]]
  return dot2add(p1, p2, p3);
}

// CHECK-LABEL: define {{.*}}test_double_arg3_type
float test_double_arg3_type(half2 p1, half2 p2, double p3) {
  // CHECK-SPIRV:  %conv = fptrunc reassoc nnan ninf nsz arcp afn double %3 to float
  // CHECK-SPIRV:  %[[RES:.*]] = call {{.*}} float @_ZN4hlsl7dot2addEDv2_DhS0_f(<2 x half> {{.*}} %1, <2 x half> {{.*}} %2, float {{.*}} %conv) #3 {{.*}}
  // CHECK-DXIL:  %conv = fptrunc reassoc nnan ninf nsz arcp afn double %2 to float
  // CHECK-DXIL:  %[[RES:.*]] = call {{.*}} float @_ZN4hlsl7dot2addEDv2_DhS0_f(<2 x half> {{.*}} %0, <2 x half> {{.*}} %1, float {{.*}} %conv) #2
  // CHECK:  ret float %[[RES]]
  return dot2add(p1, p2, p3);
}

// CHECK-LABEL: define {{.*}}test_float_arg1_arg2_type
float test_float_arg1_arg2_type(float2 p1, float2 p2, float p3) {
  // CHECK-SPIRV:  %conv = fptrunc reassoc nnan ninf nsz arcp afn <2 x float> %1 to <2 x half>
  // CHECK-SPIRV:  %conv1 = fptrunc reassoc nnan ninf nsz arcp afn <2 x float> %2 to <2 x half>
  // CHECK-SPIRV:  %[[RES:.*]] = call {{.*}} float @_ZN4hlsl7dot2addEDv2_DhS0_f(<2 x half> {{.*}} %conv, <2 x half> {{.*}} %conv1, float {{.*}} %3) #3 {{.*}}
  // CHECK-DXIL:  %conv = fptrunc reassoc nnan ninf nsz arcp afn <2 x float> %0 to <2 x half>
  // CHECK-DXIL:  %conv1 = fptrunc reassoc nnan ninf nsz arcp afn <2 x float> %1 to <2 x half>
  // CHECK-DXIL:  %[[RES:.*]] = call {{.*}} float @_ZN4hlsl7dot2addEDv2_DhS0_f(<2 x half> {{.*}} %conv, <2 x half> {{.*}} %conv1, float {{.*}} %2) #2
  // CHECK:  ret float %[[RES]]
  return dot2add(p1, p2, p3);
}

// CHECK-LABEL: define {{.*}}test_int16_arg1_arg2_type
float test_int16_arg1_arg2_type(int16_t2 p1, int16_t2 p2, float p3) {
  // CHECK-SPIRV:  %conv = sitofp <2 x i16> %1 to <2 x half>
  // CHECK-SPIRV:  %conv1 = sitofp <2 x i16> %2 to <2 x half>
  // CHECK-SPIRV:  %[[RES:.*]] = call {{.*}} float @_ZN4hlsl7dot2addEDv2_DhS0_f(<2 x half> {{.*}} %conv, <2 x half> {{.*}} %conv1, float {{.*}} %3) #3 {{.*}}
  // CHECK-DXIL:  %conv = sitofp <2 x i16> %0 to <2 x half>
  // CHECK-DXIL:  %conv1 = sitofp <2 x i16> %1 to <2 x half>
  // CHECK-DXIL:  %[[RES:.*]] = call {{.*}} float @_ZN4hlsl7dot2addEDv2_DhS0_f(<2 x half> {{.*}} %conv, <2 x half> {{.*}} %conv1, float {{.*}} %2) #2
  // CHECK:  ret float %[[RES]]
  return dot2add(p1, p2, p3);
}

// CHECK-LABEL: define {{.*}}dot2add_impl
// CHECK-SPIRV:  %[[MUL:.*]] = call reassoc nnan ninf nsz arcp afn half @llvm.spv.fdot.v2f16(<2 x half> %1, <2 x half> %2)
// CHECK-SPIRV:  %[[CONV:.*]] = fpext reassoc nnan ninf nsz arcp afn half %[[MUL]] to float
// CHECK-SPIRV:  %[[C:.*]] = load float, ptr %c.addr, align 4
// CHECK-SPIRV:  %[[RES:.*]] = fadd reassoc nnan ninf nsz arcp afn float %[[CONV]], %[[C]]
// CHECK-DXIL:  %[[RES:.*]] = call {{.*}} float @llvm.dx.dot2add.v2f16(<2 x half> %0, <2 x half> %1, float %2)
// CHECK:  ret float %[[RES]]
