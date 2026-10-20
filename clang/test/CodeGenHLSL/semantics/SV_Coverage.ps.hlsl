// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-pixel -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=CHECK-DXIL
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-pixel -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -o - %s | FileCheck %s --check-prefix=CHECK-SPIRV
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-pixel -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -DTYPE=int -o - %s | FileCheck %s --check-prefix=CHECK-DXIL

// A signed declaration still uses an unsigned (U32) signature element.
#ifndef TYPE
#define TYPE uint
#endif

// SV_Coverage is lowered to the SampleMask built-in. It must be an array of
// 32-bit integers, the coverage is the first element.
// The input is an integer fragment input, so it is also flat.
// CHECK-SPIRV-DAG: @SV_Coverage = external hidden thread_local addrspace(7) externally_initialized constant [1 x i32], !spirv.Decorations ![[#MD_IN:]]
// CHECK-SPIRV-DAG: @SV_Coverage.1 = external hidden thread_local addrspace(8) global [1 x i32], !spirv.Decorations ![[#MD_OUT:]]

// CHECK-DXIL-LABEL: define void @main()
// CHECK-SPIRV-LABEL: define void @main()
TYPE main(TYPE c : SV_Coverage) : SV_Coverage {
  // The input coverage is read with an intrinsic, not from the signature.
  // CHECK-DXIL: %[[INPUT:.*]] = call i32 @llvm.dx.coverage()
  // CHECK-DXIL: %[[RESULT:.*]] = call i32 @_Z4main{{[ij]}}(i32 %[[INPUT]])
  // CHECK-DXIL: call void @llvm.dx.store.output.i32(i32 0, i32 0, i8 0, i32 %[[RESULT]])

  // CHECK-SPIRV: %[[INPUT:.*]] = load i32, ptr addrspace(7) @SV_Coverage, align 4
  // CHECK-SPIRV: %[[RESULT:.*]] = call spir_func i32 @_Z4main{{[ij]}}(i32 %[[INPUT]])
  // CHECK-SPIRV: store i32 %[[RESULT]], ptr addrspace(8) @SV_Coverage.1, align 4
  return c & 1;
}

// CHECK-DXIL: declare i32 @llvm.dx.coverage() #[[#ATTR:]]
// CHECK-DXIL: attributes #[[#ATTR]] = { {{.*}}memory(none){{.*}} }

// SV_Coverage is only in the output signature, where it has no location.
// CHECK-DXIL: !dx.semantic.signatures = !{![[#SIG:]]}
// CHECK-DXIL: ![[#SIG]] = !{ptr @main, null, ![[#OUT:]]}
// CHECK-DXIL: ![[#OUT]] = !{![[#COV:]]}
// CHECK-DXIL: ![[#COV]] = !{i32 0, !"SV_Coverage", i32 5, i32 14, ![[#IDX:]], i32 0, i32 1, i8 1, i32 -1, i8 -1, i8 0, i8 0, i32 0}
//                                                   |      |                 |      |      |     |       `-> StartRow/StartCol: unallocated
//                                                   |      |                 |      |      |     `-> Cols
//                                                   |      |                 |      |      `-> Rows
//                                                   |      |                 |      `-> Interpolation: Undefined
//                                                   |      |                 `-> Semantic indices
//                                                   |      `-> SemanticKind: Coverage
//                                                   `-> CompType: U32
// CHECK-DXIL: ![[#IDX]] = !{i32 0}

// CHECK-SPIRV-DAG: ![[#MD_IN]] = !{![[#MD_1:]], ![[#FLAT:]]}
// CHECK-SPIRV-DAG: ![[#MD_OUT]] = !{![[#MD_1]]}
// CHECK-SPIRV-DAG: ![[#FLAT]] = !{i32 14}
// CHECK-SPIRV-DAG: ![[#MD_1]] = !{i32 11, i32 20}
//                                  |       `-> BuiltIn SampleMask
//                                  `-> SPIR-V decoration 'BuiltIn'
