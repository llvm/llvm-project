// RUN: split-file %s %t
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -ffp-contract=on -emit-pch -o %t/header.pch %t/header.hlsl
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -ffp-contract=on -include-pch %t/header.pch -emit-llvm -disable-llvm-passes -o - %t/source.hlsl | FileCheck %s --check-prefixes=CHECK,PRECISE --implicit-check-not=nofpclass
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -ffp-contract=on -include-pch %t/header.pch -emit-pch -o %t/chain.pch %t/chain.hlsl
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -ffp-contract=on -include-pch %t/chain.pch -emit-llvm -disable-llvm-passes -o - %t/source.hlsl | FileCheck %s --check-prefixes=CHECK,PRECISE --implicit-check-not=nofpclass
// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-library -ffp-contract=on -emit-pch -o %t/spirv.pch %t/header.hlsl
// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-library -ffp-contract=on -include-pch %t/spirv.pch -emit-llvm -disable-llvm-passes -o - %t/source.hlsl | FileCheck %s --check-prefixes=CHECK,PRECISE --implicit-check-not=nofpclass
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -ffp-contract=on -DNO_PRECISE -emit-pch -o %t/default.pch %t/header.hlsl
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -ffp-contract=on -DNO_PRECISE -include-pch %t/default.pch -emit-llvm -disable-llvm-passes -o - %t/source.hlsl | FileCheck %s --check-prefixes=CHECK,DEFAULT

// CHECK-LABEL: define {{.*}}float @_Z10FromHeaderff(
// PRECISE: %[[MUL:.*]] = fmul float
// PRECISE: fadd float %[[MUL]],
// DEFAULT: call {{.*}}float @llvm.fmuladd.f32(
// CHECK-LABEL: define {{.*}}float @_Z10FromSourceff(
// PRECISE: %[[MUL:.*]] = fmul float
// PRECISE: fadd float %[[MUL]],
// DEFAULT: call {{.*}}float @llvm.fmuladd.f32(

//--- header.hlsl
#ifndef NO_PRECISE
// Precise mode must be restored even if this field is never deserialized.
struct Unused {
  precise float X;
};
#endif

export float FromHeader(float A, float B) { return A * B + A; }

//--- chain.hlsl
float Declaration(float A);

//--- source.hlsl
export float FromSource(float A, float B) { return A * B + A; }
