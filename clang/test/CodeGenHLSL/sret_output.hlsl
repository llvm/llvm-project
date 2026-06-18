// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -triple dxil-pc-shadermodel6.3-library %s  \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s --check-prefixes=CHECK-DX,CHECK
// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -triple spirv-pc-vulkan-library %s  \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s --check-prefixes=CHECK-VK,CHECK

struct S {
  float a : A4;
};

// CHECK-VK: @A4 = external hidden thread_local addrspace(8) global float, !spirv.Decorations ![[#ATTR0:]]

// Make sure sret parameter is generated.
// CHECK-DX: define internal void @_Z7vs_mainv(ptr dead_on_unwind noalias writable sret(%struct.S) align 1 %agg.result)
// CHECK-VK: define internal spir_func void @_Z7vs_mainv(ptr dead_on_unwind noalias writable sret(%struct.S) align 1 %agg.result)

[shader("vertex")]
S vs_main() {
  S s;
  s.a = 0;
  return s;
};

// CHECK: %[[#alloca:]] = alloca %struct.S, align 8
// CHECK-DX:              call void @_Z7vs_mainv(ptr %[[#alloca]])
// CHECK-VK:              call spir_func void @_Z7vs_mainv(ptr %[[#alloca]])
// CHECK: %[[#a:]] = load %struct.S, ptr %[[#alloca]], align 4
// CHECK: %[[#b:]] = extractvalue %struct.S %[[#a]], 0
// CHECK-DX:         call void @llvm.dx.store.output.f32(i32 4, i32 0, i32 0, i8 0, i32 poison, float %[[#b]])
// CHECK-VK:         store float %3, ptr addrspace(8) @A4, align 4
// CHECK:            ret void

// CHECK-VK: ![[#ATTR0]] = !{![[#ATTR1:]]}
// CHECK-VK: ![[#ATTR1]] = !{i32 30, i32 0}
