// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-vertex -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK-DXIL,CHECK
// RUN: %clang_cc1 -triple spirv-linux-vulkan-vertex -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK-SPIRV,CHECK


struct Input {
  // FIXME: change this once we have a valid system semantic as input for VS.
  float Idx : B2;
};

struct Output {
  float a : A4;
  float b : A2;
};

// CHECK-SPIRV-DAG:    @B2 = external hidden thread_local addrspace(7) externally_initialized constant float, !spirv.Decorations ![[#METADATA_0:]]
// CHECK-SPIRV-DAG:    @A4 = external hidden thread_local addrspace(8) global float, !spirv.Decorations ![[#METADATA_0:]]
// CHECK-SPIRV-DAG:    @A2 = external hidden thread_local addrspace(8) global float, !spirv.Decorations ![[#METADATA_2:]]

// CHECK: %Idx = getelementptr inbounds nuw %struct.Input, ptr %input, i32 0, i32 0
// CHECK: %[[#tmp:]] = load float, ptr %Idx, align 1
// CHECK: %a = getelementptr inbounds nuw %struct.Output, ptr %agg.result, i32 0, i32 0
// CHECK: store float %[[#tmp]], ptr %a, align 1

// CHECK: %Idx1 = getelementptr inbounds nuw %struct.Input, ptr %input, i32 0, i32 0
// CHECK: %[[#tmp:]] = load float, ptr %Idx1, align 1
// CHECK: %b = getelementptr inbounds nuw %struct.Output, ptr %agg.result, i32 0, i32 1
// CHECK: store float %[[#tmp]], ptr %b, align 1

Output main(Input input) {
  Output o;
  o.a = input.Idx;
  o.b = input.Idx;
  return o;
}

// Code generated in the entrypoint wrapper:

// CHECK: %[[#OUTPUT:]] = alloca %struct.Output, align 8

// CHECK-SPIRV: call spir_func void @_Z4main5Input(ptr %[[#OUTPUT]], ptr %[[#]])
// CHECK-DXIL:  call void @_Z4main5Input(ptr %[[#OUTPUT]], ptr %[[#]])

// CHECK: %[[#TMP:]] = load %struct.Output, ptr %[[#OUTPUT]], align 4
// CHECK: %[[#VAL:]] = extractvalue %struct.Output %[[#TMP]], 0
// CHECK-SPIRV:        store float %[[#VAL]], ptr addrspace(8) @A4, align 4
// CHECK-DXIL:         call void @llvm.dx.store.output.f32(i32 4, i32 0, i32 0, i8 0, i32 poison, float %[[#VAL]])
// CHECK: %[[#VAL:]] = extractvalue %struct.Output %[[#TMP]], 1
// CHECK-SPIRV:        store float %[[#VAL]], ptr addrspace(8) @A2, align 4
// CHECK-DXIL:         call void @llvm.dx.store.output.f32(i32 4, i32 0, i32 0, i8 0, i32 poison, float %[[#VAL]])

// CHECK-SPIRV-DAG: ![[#METADATA_0]] = !{![[#METADATA_1:]]}
// CHECK-SPIRV-DAG: ![[#METADATA_2]] = !{![[#METADATA_3:]]}
// CHECK-SPIRV-DAG: ![[#METADATA_1]] = !{i32 30, i32 0}
// CHECK-SPIRV-DAG: ![[#METADATA_3]] = !{i32 30, i32 1}
//                                            |      `- Location index
//                                            `-> Decoration "Location"
