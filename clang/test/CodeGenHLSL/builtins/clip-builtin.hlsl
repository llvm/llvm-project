// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple dxil-pc-shadermodel6.3-library %s -fnative-half-type -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// CHECK:      define hidden void @{{.*}}builtin_clip_float{{.*}}(float {{.*}} [[P0:%.*]])
// CHECK:      [[LOAD:%.*]] = load float, ptr [[P0]].addr, align 4
// CHECK-NEXT: [[FCMP:%.*]] = fcmp reassoc nnan ninf nsz arcp afn olt float [[LOAD]], 0.000000e+00
// CHECK-NO:   call i1 @llvm.dx.any
// CHECK-NEXT: call void @llvm.dx.discard(i1 [[FCMP]])
void builtin_clip_float (float p0) {
  __builtin_hlsl_elementwise_clip(p0);
}
