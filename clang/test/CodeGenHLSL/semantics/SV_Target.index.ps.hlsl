// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-pixel -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=CHECK-DXIL
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-pixel -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=CHECK-SPIRV

// The semantic index of SV_Target selects the render target. An array covers
// consecutive render targets.
struct PSOut {
  float4 Color : SV_Target3;
  float4 Layers[2] : SV_Target1;
};

// In SPIR-V the location is the semantic index, not the next free location.
// CHECK-SPIRV-DAG: @SV_Target3 = external hidden thread_local addrspace(8) global <4 x float>, !spirv.Decorations ![[#LOC3:]]
// CHECK-SPIRV-DAG: @SV_Target1 = external hidden thread_local addrspace(8) global [2 x <4 x float>], !spirv.Decorations ![[#LOC1:]]

// CHECK-DXIL-LABEL: define void @main()
// CHECK-SPIRV-LABEL: define void @main()
PSOut main(float4 P : SV_Position) {
  // DXIL pixel outputs are not indexable, so the array is split into one
  // element per render target.
  // CHECK-DXIL: %[[RESULT:.*]] = load %struct.PSOut, ptr %{{.*}}, align 4
  // CHECK-DXIL: %[[COLOR:.*]] = extractvalue %struct.PSOut %[[RESULT]], 0
  // CHECK-DXIL: call void @llvm.dx.store.output.v4f32(i32 0, i32 0, i8 0, <4 x float> %[[COLOR]])
  // CHECK-DXIL: %[[LAYERS:.*]] = extractvalue %struct.PSOut %[[RESULT]], 1
  // CHECK-DXIL: %[[LAYER0:.*]] = extractvalue [2 x <4 x float>] %[[LAYERS]], 0
  // CHECK-DXIL: call void @llvm.dx.store.output.v4f32(i32 1, i32 0, i8 0, <4 x float> %[[LAYER0]])
  // CHECK-DXIL: %[[LAYER1:.*]] = extractvalue [2 x <4 x float>] %[[LAYERS]], 1
  // CHECK-DXIL: call void @llvm.dx.store.output.v4f32(i32 2, i32 0, i8 0, <4 x float> %[[LAYER1]])

  // CHECK-SPIRV: %[[RESULT:.*]] = load %struct.PSOut, ptr %{{.*}}, align 4
  // CHECK-SPIRV: %[[COLOR:.*]] = extractvalue %struct.PSOut %[[RESULT]], 0
  // CHECK-SPIRV: store <4 x float> %[[COLOR]], ptr addrspace(8) @SV_Target3, align 4
  // CHECK-SPIRV: %[[LAYERS:.*]] = extractvalue %struct.PSOut %[[RESULT]], 1
  // CHECK-SPIRV: store [2 x <4 x float>] %[[LAYERS]], ptr addrspace(8) @SV_Target1, align 4
  PSOut O;
  O.Color = P;
  O.Layers[0] = P * 2;
  O.Layers[1] = P * 3;
  return O;
}

// Each element has one row, at the row of its semantic index.
// CHECK-DXIL: !dx.semantic.signatures = !{![[#SIG:]]}
// CHECK-DXIL: ![[#SIG]] = !{ptr @main, ![[#IN:]], ![[#OUT:]]}
// CHECK-DXIL: ![[#OUT]] = !{![[#T3:]], ![[#T1:]], ![[#T2:]]}
// CHECK-DXIL: ![[#T3]] = !{i32 0, !"SV_Target", i32 9, i32 16, ![[#IDX3:]], i32 0, i32 1, i8 4, i32 3, i8 0, i8 0, i8 0, i32 0}
//                                                 |      |                 |      |      |     |       `-> StartRow: semantic index
//                                                 |      |                 |      |      |     `-> Cols
//                                                 |      |                 |      |      `-> Rows
//                                                 |      |                 |      `-> Interpolation: Undefined
//                                                 |      |                 `-> Semantic indices
//                                                 |      `-> SemanticKind: Target
//                                                 `-> CompType: F32
// CHECK-DXIL: ![[#IDX3]] = !{i32 3}
// CHECK-DXIL: ![[#T1]] = !{i32 1, !"SV_Target", i32 9, i32 16, ![[#IDX1:]], i32 0, i32 1, i8 4, i32 1, i8 0, i8 0, i8 0, i32 0}
// CHECK-DXIL: ![[#IDX1]] = !{i32 1}
// CHECK-DXIL: ![[#T2]] = !{i32 2, !"SV_Target", i32 9, i32 16, ![[#IDX2:]], i32 0, i32 1, i8 4, i32 2, i8 0, i8 0, i8 0, i32 0}
// CHECK-DXIL: ![[#IDX2]] = !{i32 2}

// CHECK-SPIRV-DAG: ![[#LOC3]] = !{![[#LOC3_MD:]]}
// CHECK-SPIRV-DAG: ![[#LOC3_MD]] = !{i32 30, i32 3}
// CHECK-SPIRV-DAG: ![[#LOC1]] = !{![[#LOC1_MD:]]}
// CHECK-SPIRV-DAG: ![[#LOC1_MD]] = !{i32 30, i32 1}
//                                        |       `-> Location index
//                                        `-> SPIR-V decoration 'Location'
