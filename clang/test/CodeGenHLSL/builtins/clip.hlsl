// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.3-pixel %s -fnative-half-type -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -finclude-default-header -triple spirv-vulkan-pixel %s -fnative-half-type -emit-llvm -o - | FileCheck %s --check-prefix=SPIRV


void test_scalar(float Buf) {
  // CHECK:      define void @{{.*}}test_scalar{{.*}}(float {{.*}} [[VALP:%.*]])
  // CHECK:      [[LOAD:%.*]] = load float, ptr [[VALP]].addr, align 4
  // CHECK-NEXT: [[FCMP:%.*]] = fcmp reassoc nnan ninf nsz arcp afn olt float [[LOAD]], 0.000000e+00
  // CHECK-NO:   call i1 @llvm.dx.any
  // CHECK-NEXT: call void @llvm.dx.discard(i1 [[FCMP]])
  //
  // SPIRV:      define spir_func void @{{.*}}test_scalar{{.*}}(float {{.*}} [[VALP:%.*]])
  // SPIRV:      [[LOAD:%.*]] = load float, ptr [[VALP]].addr, align 4
  // SPIRV-NEXT: [[FCMP:%.*]] = fcmp reassoc nnan ninf nsz arcp afn olt float [[LOAD]], 0.000000e+00
  // SPIRV-NO:   call i1 @llvm.spv.any
  // SPIRV-NEXT: br i1 [[FCMP]], label %[[LTL:.*]], label %[[ENDL:.*]]
  // SPIRV:      [[LTL]]: ; preds = %entry
  // SPIRV-NEXT: call void @llvm.spv.discard()
  // SPIRV:      br label %[[ENDL]]
  clip(Buf);
}

void test_vector4(float4 Buf) {
  // CHECK:      define void @{{.*}}test_vector{{.*}}(<4 x float> {{.*}} [[VALP:%.*]])
  // CHECK:      [[LOAD:%.*]] = load <4 x float>, ptr [[VALP]].addr, align 16
  // CHECK-NEXT: [[FCMP:%.*]] = fcmp reassoc nnan ninf nsz arcp afn olt <4 x float> [[LOAD]], zeroinitializer
  // CHECK-NEXT: [[ANYC:%.*]] = call i1 @llvm.dx.any.v4i1(<4 x i1> [[FCMP]])
  // CHECK-NEXT: call void @llvm.dx.discard(i1 [[ANYC]])
  //
  // SPIRV:      define spir_func void @{{.*}}test_vector{{.*}}(<4 x float> {{.*}} [[VALP:%.*]])
  // SPIRV:      [[LOAD:%.*]] = load <4 x float>, ptr [[VALP]].addr, align 16
  // SPIRV-NEXT: [[FCMP:%.*]] = fcmp reassoc nnan ninf nsz arcp afn olt <4 x float> [[LOAD]], zeroinitializer
  // SPIRV-NEXT: [[ANYC:%.*]] = call i1 @llvm.spv.any.v4i1(<4 x i1> [[FCMP]]) 
  // SPIRV-NEXT: br i1 [[ANYC]], label %[[LTL:.*]], label %[[ENDL:.*]]
  // SPIRV:      [[LTL]]: ; preds = %entry
  // SPIRV-NEXT: call void @llvm.spv.discard()
  // SPIRV-NEXT: br label %[[ENDL]]
  clip(Buf);
}
