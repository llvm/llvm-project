// RUN: %clang_cc1 -triple dxil-unknown-shadermodel6.8-vertex -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck --check-prefix=CHECK-DXIL %s
// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-vertex -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck --check-prefix=CHECK-SPIRV  %s

// CHECK-SPIRV: @SV_VertexID = external hidden thread_local addrspace(7) externally_initialized constant i32, !spirv.Decorations ![[#MD_0:]]

// CHECK: define void @main() {{.*}} {
uint main(uint id : SV_VertexID) : A {
  // CHECK-SPIRV: %[[#P:]] = load i32, ptr addrspace(7) @SV_VertexID, align 4
  // CHECK-SPIRV:   %[[#]] = call spir_func i32 @_Z4mainj(i32 %[[#P]])

  // CHECK-DXIL: %SV_VertexID0 = call i32 @llvm.dx.load.input.i32(i32 4, i32 0, i32 0, i8 0, i32 poison)
  // CHECK-DXIL:        %[[#]] = call i32 @_Z4mainj(i32 %SV_VertexID0)
  return id;
}

// CHECK-SPIRV-DAG: ![[#MD_0]] = !{![[#MD_1:]]}
// CHECK-SPIRV-DAG: ![[#MD_1]] = !{i32 11, i32 42}
//                                      |       `-> BuiltIn VertexIndex
//                                      `-> SPIR-V decoration 'BuiltIn'
