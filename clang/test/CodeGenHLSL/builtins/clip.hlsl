// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-pixel %s -fnative-half-type -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple spirv-vulkan-pixel %s -fnative-half-type -emit-llvm -o - | FileCheck %s --check-prefix=SPIRV


void test_scalar(float Buf) {
  // CHECK:      define void @{{.*}}test_scalar{{.*}}(float {{.*}} [[VALP:%.*]])
  // CHECK:      [[LOAD:%.*]] = load float, ptr [[VALP]].addr, align 4
  // CHECK-NEXT: [[FCMP:%.*]] = fcmp olt float [[LOAD]], 0.000000e+00
  // CHECK-NEXT: call void @llvm.dx.clip.i1(i1 [[FCMP]])
  //
  // SPIRV:      define spir_func void @{{.*}}test_scalar{{.*}}(float {{.*}} [[VALP:%.*]])
  // SPIRV:      [[LOAD:%.*]] = load float, ptr [[VALP]].addr, align 4
  // SPIRV-NEXT: [[FCMP:%.*]] = fcmp olt float [[LOAD]], 0.000000e+00
  // SPIRV-NEXT: br i1 [[FCMP]], label %[[LTL:.*]], label %[[ENDL:.*]]
  // SPIRV: [[LTL]]: ; preds = %entry
  // SPIRV-NEXT: call void @llvm.spv.clip()
  // SPIRV: br label %[[ENDL]]
  clip(Buf);
}

void test_vector4(float4 Buf) {
  // CHECK:      define void @{{.*}}test_vector{{.*}}(<4 x float> {{.*}} [[VALP:%.*]])
  // CHECK:      [[LOAD:%.*]] = load <4 x float>, ptr [[VALP]].addr, align 16
  // CHECK-NEXT: [[FCMP:%.*]] = fcmp olt <4 x float> [[LOAD]], zeroinitializer
  // CHECK-NEXT: call void @llvm.dx.clip.v4i1(<4 x i1> [[FCMP]])
  //
  // SPIRV:      define spir_func void @{{.*}}test_vector{{.*}}(<4 x float> {{.*}} [[VALP:%.*]])
  // SPIRV:      [[LOAD:%.*]] = load <4 x float>, ptr [[VALP]].addr, align 16
  // SPIRV-NEXT: [[FCMP:%.*]] = fcmp olt <4 x float> [[LOAD]], zeroinitializer
  // SPIRV-NEXT: [[RED:%.*]] = call i1 @llvm.vector.reduce.or.v4i1(<4 x i1> [[FCMP]])
  // SPIRV-NEXT: br i1 [[RED]], label %[[LTL:.*]], label %[[ENDL:.*]]
  // SPIRV: [[LTL]]: ; preds = %entry
  // SPIRV-NEXT: call void @llvm.spv.clip()
  // SPIRV-NEXT: br label %[[ENDL]]
  clip(Buf);
}
