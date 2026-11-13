// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-pixel -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -DTYPE=uint -o - %s | FileCheck %s --check-prefixes=CHECK-DXIL
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-pixel -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -DTYPE=uint -o - %s | FileCheck %s --check-prefixes=CHECK-SPIRV
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-pixel -x hlsl -emit-llvm -finclude-default-header -disable-llvm-passes -DTYPE=bool -o - %s | FileCheck %s --check-prefixes=CHECK-SPIRV

// SV_IsFrontFace as a member of an entry point input struct. The struct member
// is stored as a 32-bit integer, so SPIR-V converts the bool FrontFacing
// variable to i32 for both a uint and a bool member.
//
// A bool member is only tested for SPIR-V. The DXIL CodeGen path asserts on any
// bool struct member that has a semantic, which is a separate bug.

struct S {
  float4 pos : SV_Position;
  TYPE ff : SV_IsFrontFace;
};

// CHECK-SPIRV-LABEL: define void @main()
// CHECK-DXIL-LABEL: define void @main()
float4 main(S s) : SV_Target {
  // CHECK-SPIRV: %[[FF:.*]] = load i1, ptr addrspace(7) @SV_IsFrontFace
  // CHECK-SPIRV: %[[EXT:.*]] = zext i1 %[[FF]] to i32

  // CHECK-DXIL: call i32 @llvm.dx.load.input.i32(i32 1, i32 0, i8 0, i32 poison)
  return s.ff ? s.pos : float4(0, 0, 0, 0);
}
