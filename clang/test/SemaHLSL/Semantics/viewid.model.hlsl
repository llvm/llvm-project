// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -finclude-default-header -x hlsl -verify=dxil -o - %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.1-library -finclude-default-header -x hlsl -verify=ok -o - %s
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.2-library -finclude-default-header -x hlsl -verify=ok -o - %s
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-library -finclude-default-header -x hlsl -verify=ok -o - %s

// SV_ViewID requires shader model 6.1 for DXIL.
// For SPIR-V, SV_ViewID is the ViewIndex built-in, which requires the
// MultiView capability. MultiView is core from Vulkan 1.1, and Clang only
// accepts Vulkan 1.2 or later, so every supported Vulkan version can use it.
// The SPIR-V runs only check that no error is emitted. The SPV_KHR_multiview
// requirement before SPIR-V 1.3 is covered by
// llvm/test/CodeGen/SPIRV/semantics/view-index.ps.ll.

[shader("pixel")]
float4 ps_in(uint v : SV_ViewID) : SV_Target {
// dxil-error@-1 {{semantic 'SV_ViewID' requires shader model 6.1 or greater}}
  return float4((float)v, 0, 0, 1);
}

// ok-no-diagnostics
