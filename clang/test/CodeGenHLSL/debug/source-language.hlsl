// Verify that HLSL shaders are tagged with DW_LANG_HLSL in the debug compile
// unit. For DWARFv6, verify the sourceLanguageName field uses DW_LNAME_HLSL.

// DXIL target, DWARFv4
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-compute -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -hlsl-entry main \
// RUN:   -debug-info-kind=standalone -dwarf-version=4 -o - %s \
// RUN:   | FileCheck %s --check-prefix=CHECK-V4

// SPIR-V target, DWARFv4
// RUN: %clang_cc1 -triple spirv-unknown-vulkan-compute -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -hlsl-entry main \
// RUN:   -debug-info-kind=standalone -dwarf-version=4 -o - %s \
// RUN:   | FileCheck %s --check-prefix=CHECK-V4

// DXIL target, DWARFv6
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-compute -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -hlsl-entry main \
// RUN:   -debug-info-kind=standalone -dwarf-version=6 -o - %s \
// RUN:   | FileCheck %s --check-prefix=CHECK-V6

// SPIR-V target, DWARFv6
// RUN: %clang_cc1 -triple spirv-unknown-vulkan-compute -x hlsl -emit-llvm \
// RUN:   -disable-llvm-passes -hlsl-entry main \
// RUN:   -debug-info-kind=standalone -dwarf-version=6 -o - %s \
// RUN:   | FileCheck %s --check-prefix=CHECK-V6

// CHECK-V4: !DICompileUnit(language: DW_LANG_HLSL,
// CHECK-V4-NOT: !DICompileUnit(language: DW_LANG_C_plus_plus

// CHECK-V6: !DICompileUnit(sourceLanguageName: DW_LNAME_HLSL,
// CHECK-V6-NOT: !DICompileUnit(sourceLanguageName: DW_LNAME_C_plus_plus

[numthreads(1, 1, 1)] void main() {}
