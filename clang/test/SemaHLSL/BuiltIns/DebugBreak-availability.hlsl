// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.9-library \
// RUN:   -fsyntax-only -verify %s
// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.0-compute \
// RUN:   -fsyntax-only -verify %s
// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.10-compute \
// RUN:   -fsyntax-only -verify=available %s
// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.10-library \
// RUN:   -fsyntax-only -verify=available %s
// RUN: %clang_cc1 -finclude-default-header -triple spirv-unknown-vulkan-library \
// RUN:   -fsyntax-only -verify=available %s

// available-no-diagnostics

[shader("compute")]
[numthreads(1, 1, 1)]
void main() {

  DebugBreak();
  // expected-error@-1 {{'DebugBreak' is only available on Shader Model 6.10 or newer}}
  // expected-note@hlsl/hlsl_debugging.h:* {{'DebugBreak' has been marked as being introduced in Shader Model 6.10 here}}

  hlsl::DebugBreak();
  // expected-error@-1 {{'DebugBreak' is only available on Shader Model 6.10 or newer}}
  // expected-note@hlsl/hlsl_debugging.h:* {{'DebugBreak' has been marked as being introduced in Shader Model 6.10 here}}
}
