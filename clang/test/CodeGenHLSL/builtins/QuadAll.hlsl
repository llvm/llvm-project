// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   dxil-pc-shadermodel6.7-library %s -emit-llvm \
// RUN:   -disable-llvm-passes -o - | FileCheck %s --check-prefixes=CHECK,DXIL
// RUN: %clang_cc1 -finclude-default-header -triple \
// RUN:   spirv-pc-vulkan-library %s -emit-llvm -disable-llvm-passes -o - | \
// RUN:   FileCheck %s --check-prefixes=CHECK,SPIRV

// CHECK-LABEL: define {{.*}}test_quad_all{{.*}}(
// SPIRV: %[[TOKEN:.*]] = call token @llvm.experimental.convergence.entry()
// DXIL: %[[RET:.*]] = call i1 @llvm.dx.quad.all(i1 %{{.*}})
// SPIRV: %[[RET:.*]] = call i1 @llvm.spv.quad.all(i1 %{{.*}}) [ "convergencectrl"(token %[[TOKEN]]) ]
// CHECK: ret i1 %[[RET]]
bool test_quad_all(bool cond) {
  return QuadAll(cond);
}

// CHECK-LABEL: define {{.*}}test_builtin{{.*}}(
// SPIRV: %[[TOKEN:.*]] = call token @llvm.experimental.convergence.entry()
// DXIL: %[[RET:.*]] = call i1 @llvm.dx.quad.all(i1 %{{.*}})
// SPIRV: %[[RET:.*]] = call i1 @llvm.spv.quad.all(i1 %{{.*}}) [ "convergencectrl"(token %[[TOKEN]]) ]
// CHECK: ret i1 %[[RET]]
bool test_builtin(bool cond) {
  return __builtin_hlsl_quad_all(cond);
}
