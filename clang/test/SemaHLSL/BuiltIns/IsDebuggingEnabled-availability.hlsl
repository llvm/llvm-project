// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.9-library \
// RUN:   -fsyntax-only -verify %s
// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.0-compute \
// RUN:   -fsyntax-only -verify %s
// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.10-compute \
// RUN:   -fsyntax-only -verify=available %s
// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.10-compute \
// RUN:   -fsyntax-only -verify=available %s
// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.10-library \
// RUN:   -fsyntax-only -verify=available %s
// RUN: %clang_cc1 -finclude-default-header -triple spirv-unknown-vulkan-library \
// RUN:   -fsyntax-only -verify=spirv %s

// available-no-diagnostics

[shader("compute")]
[numthreads(1, 1, 1)]
void main() {
  // expected-error@+3 {{'IsDebuggingEnabled' is only available on Shader Model 6.10 or newer}}
  // expected-note@hlsl/hlsl_debugging.h:* {{'IsDebuggingEnabled' has been marked as being introduced in Shader Model 6.10 here}}
  // spirv-error@+1 {{no member named 'dx' in namespace 'hlsl'}}
  bool a = hlsl::dx::IsDebuggingEnabled();

  // spirv-error@+1 {{builtin is not supported on this target}}
  bool b = __builtin_hlsl_isdebuggingenabled();
}
