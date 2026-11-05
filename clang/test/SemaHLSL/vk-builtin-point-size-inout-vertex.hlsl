// RUN: %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan1.3-vertex -hlsl-entry vs -fsyntax-only -verify %s

// An `inout` PointSize parameter on a vertex shader is accepted, same as
// plain `out` (see the sibling vk-builtin-point-size-inout-pixel.hlsl for a
// stage where `inout`'s current treatment as output-only changes the
// outcome).

// SPIRV-VAL-SKIP: inout entry parameters are not supported by clang's SPIR-V
// CodeGen yet (llvm_unreachable "Not handled yet" in CGHLSLRuntime.cpp). This
// test only covers Sema.
[shader("vertex")]
float4 vs([[vk::builtin("PointSize")]] inout float size : PSIZE) : SV_Position {
  size *= 2.0;
  return float4(0, 0, 0, 1);
}

// expected-no-diagnostics
