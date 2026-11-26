// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -finclude-default-header -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple spirv-pc-vulkan1.3-library -finclude-default-header -fsyntax-only -verify %s

// Valid type, index and interpolation uses of SV_IsFrontFace. The invalid
// cases are covered by isfrontface.ps.errors.hlsl, and the shader stage and IO
// direction by isfrontface.stages.hlsl.

// expected-no-diagnostics

// A bool or a 32-bit integer, signed or unsigned, is accepted. The SPIR-V
// FrontFacing variable is always bool, and CodeGen converts it to the declared
// type.
[shader("pixel")]
void valid_bool(bool s : SV_IsFrontFace) {}
[shader("pixel")]
void valid_uint(uint s : SV_IsFrontFace) {}
[shader("pixel")]
void valid_int(int s : SV_IsFrontFace) {}
// The semantic name is case insensitive, and index 0 is the same as no index.
[shader("pixel")]
void valid_lowercase(bool s : sv_isfrontface) {}
[shader("pixel")]
void valid_index(bool s : SV_IsFrontFace0) {}
[shader("pixel")]
void valid_flat_bool(nointerpolation bool s : SV_IsFrontFace) {}
[shader("pixel")]
void valid_flat_uint(nointerpolation uint s : SV_IsFrontFace) {}

struct Input {
  bool s : SV_IsFrontFace;
};
[shader("pixel")]
void valid_struct(Input i) {}
