// RUN: %clang_cc1 -triple spirv-linux-vulkan-vertex -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s

struct Inner {
    uint b : B;
};

struct VSInput {
    float4 position : POSITION; // Not system semantic, Location 0
};

struct VSOutput {
    float4 position : SV_POSITION; // System semantic, builtin Position
    uint a : A;  // Location 0
    Inner inner; // Location 1
    uint c : C;  // Location 2
};


// CHECK: @POSITION0 = external hidden thread_local addrspace(7) externally_initialized constant <4 x float>, !spirv.Decorations ![[#MD_0:]]
// CHECK: @SV_POSITION = external hidden thread_local addrspace(8) global <4 x float>, !spirv.Decorations ![[#MD_2:]]
// CHECK: @A0 = external hidden thread_local addrspace(8) global i32, !spirv.Decorations ![[#MD_0:]]
// CHECK: @B0 = external hidden thread_local addrspace(8) global i32, !spirv.Decorations ![[#MD_4:]]
// CHECK: @C0 = external hidden thread_local addrspace(8) global i32, !spirv.Decorations ![[#MD_6:]]

VSOutput main(VSInput input) {
    VSOutput output;
    output.position = input.position;
    output.a = 1;
    output.inner.b = 2;
    output.c = 3;
    return output;
}

// CHECK: ![[#MD_0]] = !{![[#MD_1:]]}
// CHECK: ![[#MD_1]] = !{i32 30, i32 0}
//                            |       `-> Location index
//                            `-> SPIR-V decoration 'Location'

// CHECK: ![[#MD_2]] = !{![[#MD_3:]]}
// CHECK: ![[#MD_3]] = !{i32 11, i32 0}
//                            |       `-> BuiltIn 'Position'
//                            `-> SPIR-V decoration 'BuiltIn'

// CHECK: ![[#MD_4]] = !{![[#MD_5:]]}
// CHECK: ![[#MD_5]] = !{i32 30, i32 1}
//                            |       `-> Location index
//                            `-> SPIR-V decoration 'Location'

// CHECK: ![[#MD_6]] = !{![[#MD_7:]]}
// CHECK: ![[#MD_7]] = !{i32 30, i32 2}
//                            |       `-> Location index
//                            `-> SPIR-V decoration 'Location'
