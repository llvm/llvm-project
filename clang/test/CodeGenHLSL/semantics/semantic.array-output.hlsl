// RUN: %clang_cc1 -triple spirv-linux-vulkan-library -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-SPIRV -DTARGET=spv

struct S0 {
  float4 position[2];
  float4 color;
};

// CHECK: %struct.S0 = type { [2 x <4 x float>], <4 x float> }

// CHECK-SPIRV: @A0 = external hidden thread_local addrspace(7) externally_initialized constant <4 x float>, !spirv.Decorations ![[#MD_0:]]
// CHECK-SPIRV: @B0 = external hidden thread_local addrspace(8) global [2 x <4 x float>], !spirv.Decorations ![[#MD_0:]]
// CHECK-SPIRV: @B2 = external hidden thread_local addrspace(8) global <4 x float>, !spirv.Decorations ![[#MD_2:]]

// CHECK:       define void @main0()
// CHECK:        %[[#OUT:]] = alloca %struct.S0, align 16
// CHECK-SPIRV:  %[[#ARG:]] = load <4 x float>, ptr addrspace(7) @A0, align 16
// CHECK-SPIRV:               call spir_func void @{{.*}}main0{{.*}}(ptr %[[#OUT]], <4 x float> %[[#ARG]])
// CHECK:        %[[#TMP:]] = load %struct.S0, ptr %[[#OUT]], align 16
// CHECK:         %[[#B0:]] = extractvalue %struct.S0 %[[#TMP]], 0
// CHECK-SPIRV:               store [2 x <4 x float>] %4, ptr addrspace(8) @B0, align 16
// CHECK:         %[[#B2:]] = extractvalue %struct.S0 %[[#TMP]], 1
// CHECK-SPIRV:               store <4 x float> %5, ptr addrspace(8) @B2, align 16
[shader("vertex")]
S0 main0(float4 input : A) : B {
  S0 output;
  output.position[0] = input;
  output.position[1] = input;
  output.color = input;
  return output;
}

// CHECK-SPIRV: ![[#MD_0]] = !{![[#MD_1:]]}
// CHECK-SPIRV: ![[#MD_1]] = !{i32 30, i32 0}
// CHECK-SPIRV: ![[#MD_2]] = !{![[#MD_3:]]}
// CHECK-SPIRV: ![[#MD_3]] = !{i32 30, i32 2}
