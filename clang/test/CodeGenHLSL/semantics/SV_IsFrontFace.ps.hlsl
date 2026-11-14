// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-pixel -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -DTYPE=bool -o - %s | FileCheck -DMANGLE=b %s --check-prefixes=CHECK-DXIL,CHECK-DXIL-BOOL,CHECK-DXIL-U32
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-pixel -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -DTYPE=uint -o - %s | FileCheck -DMANGLE=j %s --check-prefixes=CHECK-DXIL,CHECK-DXIL-INT,CHECK-DXIL-U32
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-pixel -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -DTYPE=int -o - %s | FileCheck -DMANGLE=i %s --check-prefixes=CHECK-DXIL,CHECK-DXIL-INT,CHECK-DXIL-I32
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-pixel -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -DTYPE=bool -o - %s | FileCheck -DMANGLE=b %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-BOOL
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-pixel -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -DTYPE=uint -o - %s | FileCheck -DMANGLE=j %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-INT
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-pixel -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -DTYPE=int -o - %s | FileCheck -DMANGLE=i %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-INT
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-pixel -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -DTYPE=bool -DMODIFIER=nointerpolation -o - %s | FileCheck -DMANGLE=b %s --check-prefixes=CHECK-DXIL,CHECK-DXIL-BOOL,CHECK-DXIL-U32
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-pixel -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -DTYPE=bool -DMODIFIER=nointerpolation -o - %s | FileCheck -DMANGLE=b %s --check-prefixes=CHECK-SPIRV,CHECK-SPIRV-BOOL

// SV_IsFrontFace accepts a bool, a uint or an int. DXIL carries a
// bool or uint as U32 (5) and an int as I32 (4). The SPIR-V FrontFacing
// variable is always a bool, and an integer parameter is converted from it.
// The last two RUN lines add the one legal interpolation modifier,
// nointerpolation, which must leave the output unchanged.

// CHECK-SPIRV: @SV_IsFrontFace = external hidden thread_local addrspace(7) externally_initialized constant i1, !spirv.Decorations ![[#MD_0:]]

// CHECK-DXIL-LABEL: define void @main()
// CHECK-SPIRV-LABEL: define void @main()
#ifndef MODIFIER
#define MODIFIER
#endif

float4 main(MODIFIER TYPE ff : SV_IsFrontFace) : SV_Target {
  // CHECK-DXIL: %[[INPUT:.*]] = call i32 @llvm.dx.load.input.i32(i32 0, i32 0, i8 0, i32 poison)
  // CHECK-DXIL-BOOL: %[[BOOL:.*]] = icmp ne i32 %[[INPUT]], 0
  // CHECK-DXIL-BOOL: call <4 x float> @_Z4main[[MANGLE]](i1 %[[BOOL]])
  // CHECK-DXIL-INT: call <4 x float> @_Z4main[[MANGLE]](i32 %[[INPUT]])

  // CHECK-SPIRV: %[[FF:.*]] = load i1, ptr addrspace(7) @SV_IsFrontFace
  // CHECK-SPIRV-BOOL: call spir_func <4 x float> @_Z4main[[MANGLE]](i1 %[[FF]])
  // CHECK-SPIRV-INT: %[[EXT:.*]] = zext i1 %[[FF]] to i32
  // CHECK-SPIRV-INT: call spir_func <4 x float> @_Z4main[[MANGLE]](i32 %[[EXT]])
  return ff ? float4(1, 0, 0, 1) : float4(0, 0, 1, 1);
}

// CHECK-DXIL-U32-DAG: !{i32 0, !"SV_IsFrontFace", i32 5, i32 13, ![[#ZERO:]], i32 1, i32 1, i8 1, i32 0, i8 0, i8 0, i8 0, i32 0}
// CHECK-DXIL-I32-DAG: !{i32 0, !"SV_IsFrontFace", i32 4, i32 13, ![[#ZERO:]], i32 1, i32 1, i8 1, i32 0, i8 0, i8 0, i8 0, i32 0}
// CHECK-DXIL-DAG: ![[#ZERO]] = !{i32 0}

// Two decorations in one node: BuiltIn FrontFacing (17) and Flat (14).
// CHECK-SPIRV-DAG: ![[#MD_0]] = !{![[#MD_1:]], ![[#MD_2:]]}
// CHECK-SPIRV-DAG: ![[#MD_1]] = !{i32 11, i32 17}
// CHECK-SPIRV-DAG: ![[#MD_2]] = !{i32 14}
