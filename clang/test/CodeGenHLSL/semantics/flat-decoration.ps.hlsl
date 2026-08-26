// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-pixel -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=CHECK-SPIRV

// CHECK-SPIRV-DAG:  @A0 = external hidden thread_local addrspace(7) externally_initialized constant float, !spirv.Decorations ![[#FLOAT_MD:]]
// CHECK-SPIRV-DAG:  @B0 = external hidden thread_local addrspace(7) externally_initialized constant i32, !spirv.Decorations ![[#INT_MD:]]
// CHECK-SPIRV-DAG:  @C0 = external hidden thread_local addrspace(7) externally_initialized constant double, !spirv.Decorations ![[#DOUBLE_MD:]]
// CHECK-SPIRV-DAG:  @D0 = external hidden thread_local addrspace(7) externally_initialized constant <2 x i32>, !spirv.Decorations ![[#IVEC_MD:]]

float main(float a : A, int b : B, double c : C, int2 d : D) : SV_Target {
  return a + b + (float)c + d.x + d.y;
}

// The float input only carries a Location decoration (no Flat).
// CHECK-SPIRV-DAG: ![[#FLOAT_MD]] = !{![[#FLOAT_LOC:]]}
// CHECK-SPIRV-DAG: ![[#FLOAT_LOC]] = !{i32 30, i32 0}

// The integer input carries both Location and Flat decorations in one node.
// CHECK-SPIRV-DAG: ![[#INT_MD]] = !{![[#INT_LOC:]], ![[#FLAT:]]}
// CHECK-SPIRV-DAG: ![[#INT_LOC]] = !{i32 30, i32 1}

// The double input carries both Location and Flat decorations in one node.
// CHECK-SPIRV-DAG: ![[#DOUBLE_MD]] = !{![[#DOUBLE_LOC:]], ![[#FLAT]]}
// CHECK-SPIRV-DAG: ![[#DOUBLE_LOC]] = !{i32 30, i32 2}

// The integer vector input carries both Location and Flat decorations in one node.
// CHECK-SPIRV-DAG: ![[#IVEC_MD]] = !{![[#IVEC_LOC:]], ![[#FLAT]]}
// CHECK-SPIRV-DAG: ![[#IVEC_LOC]] = !{i32 30, i32 3}

// CHECK-SPIRV-DAG: ![[#FLAT]] = !{i32 14}
//                                     `-> SPIR-V decoration 'Flat'
