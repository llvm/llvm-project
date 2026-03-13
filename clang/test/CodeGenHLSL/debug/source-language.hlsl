// Verify that HLSL shaders are tagged with DW_LANG_HLSL in the debug compile
// unit.

// DXIL target
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-compute -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -hlsl-entry main \
// RUN:   -debug-info-kind=standalone -dwarf-version=4 -o - %s \
// RUN:   | FileCheck %s

// SPIR-V target
// RUN: %clang_cc1 -triple spirv-unknown-vulkan-compute -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -hlsl-entry main \
// RUN:   -debug-info-kind=standalone -dwarf-version=4 -o - %s \
// RUN:   | FileCheck %s

// CHECK: !DICompileUnit(language: DW_LANG_HLSL,
// CHECK-NOT: !DICompileUnit(language: DW_LANG_C_plus_plus

[numthreads(1, 1, 1)] void main() {}
