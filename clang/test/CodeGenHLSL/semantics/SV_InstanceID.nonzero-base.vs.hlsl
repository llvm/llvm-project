// With -fvk-support-nonzero-base-instance, SV_InstanceID is
// InstanceIndex - BaseInstance. Without it, SV_InstanceID is a raw load of
// InstanceIndex (see SV_InstanceID.vs.hlsl).

// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-vertex -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -fvk-support-nonzero-base-instance -o - %s | FileCheck %s --check-prefix=CHECK-FLAG
// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-vertex -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=CHECK-NOFLAG

// CHECK-FLAG: @SV_InstanceID = external hidden thread_local addrspace(7) externally_initialized constant i32, !spirv.Decorations ![[#MD_0:]]
// CHECK-FLAG: @SV_InstanceID.base = external hidden thread_local addrspace(7) externally_initialized constant i32, !spirv.Decorations ![[#MD_BASE:]]

// CHECK-NOFLAG: @SV_InstanceID = external hidden thread_local addrspace(7) externally_initialized constant i32
// CHECK-NOFLAG-NOT: @SV_InstanceID.base

// CHECK-FLAG: define void @main() {{.*}} {
// CHECK-NOFLAG: define void @main() {{.*}} {
uint main(uint id : SV_InstanceID) : A {
  // CHECK-FLAG: %[[INDEX:.*]] = load i32, ptr addrspace(7) @SV_InstanceID, align 4
  // CHECK-FLAG: %[[BASE:.*]] = load i32, ptr addrspace(7) @SV_InstanceID.base, align 4
  // CHECK-FLAG: %[[INPUT:.*]] = sub i32 %[[INDEX]], %[[BASE]]
  // CHECK-FLAG: %[[RESULT:.*]] = call spir_func i32 @_Z4mainj(i32 %[[INPUT]])

  // CHECK-NOFLAG: %[[INPUT:.*]] = load i32, ptr addrspace(7) @SV_InstanceID, align 4
  // CHECK-NOFLAG-NOT: sub i32
  // CHECK-NOFLAG: %[[RESULT:.*]] = call spir_func i32 @_Z4mainj(i32 %[[INPUT]])
  return id;
}

// CHECK-FLAG-DAG: ![[#MD_0]] = !{![[#MD_1:]]}
// CHECK-FLAG-DAG: ![[#MD_1]] = !{i32 11, i32 43}
//                                     |       `-> BuiltIn InstanceIndex
//                                     `-> SPIR-V decoration 'BuiltIn'
// CHECK-FLAG-DAG: ![[#MD_BASE]] = !{![[#MD_BASE_1:]]}
// CHECK-FLAG-DAG: ![[#MD_BASE_1]] = !{i32 11, i32 4425}
//                                          |       `-> BuiltIn BaseInstance
//                                          `-> SPIR-V decoration 'BuiltIn'
