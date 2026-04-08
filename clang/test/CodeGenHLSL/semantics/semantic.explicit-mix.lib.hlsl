// RUN: %clang_cc1 -triple spirv-linux-vulkan-library -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK

// The followiong file contains both implicit and explicit vk::location, but
// because each entrypoint has only one kind, this is allowed.

[shader("vertex")]
float4 vs_main(float4 p : SV_Position) : A {
  return p;
}

[shader("pixel")]
float4 ps_main([[vk::location(0)]] float4 p : A) : SV_Target {
  return p;
}

// The following function is not marked as being a shader entrypoint, this
// means the semantics and [[vk::location]] attributes are ignored.
// Otherwise, the partial explicit location assignment would be illegal.
float4 not_an_entry([[vk::location(0)]] float4 a : A, float4 b : B) : C {
  return a + b;
}

// CHECK: @SV_Position0 = external hidden thread_local addrspace(7) externally_initialized constant <4 x float>, !spirv.Decorations ![[#MD_0:]]
// CHECK: @A0 = external hidden thread_local addrspace(8) global <4 x float>, !spirv.Decorations ![[#MD_0:]]
// CHECK: @A0.1 = external hidden thread_local addrspace(7) externally_initialized constant <4 x float>, !spirv.Decorations ![[#MD_0:]]
// CHECK: @SV_Target0 = external hidden thread_local addrspace(8) global <4 x float>, !spirv.Decorations ![[#MD_2:]]


// CHECK: define void @vs_main()
// CHECK: %[[#]] = load <4 x float>, ptr addrspace(7) @SV_Position0, align 16
// CHECK: store <4 x float> %[[#]], ptr addrspace(8) @A0, align 16

// CHECK: define void @ps_main()
// CHECK: %[[#]] = load <4 x float>, ptr addrspace(7) @A0.1, align 16
// CHECK: store <4 x float> %[[#]], ptr addrspace(8) @SV_Target0, align 16

// CHECK: ![[#MD_0]] = !{![[#MD_1:]]}
// CHECK: ![[#MD_1]] = !{i32 30, i32 0}
// CHECK: ![[#MD_2]] = !{![[#MD_3:]]}
// CHECK: ![[#MD_3]] = !{i32 30, i32 1}
