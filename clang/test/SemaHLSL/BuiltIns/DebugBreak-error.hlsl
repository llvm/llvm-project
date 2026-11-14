// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.10-library \
// RUN:   -fsyntax-only -verify %s
// RUN: %clang_cc1 -finclude-default-header -triple spirv-unknown-vulkan-library \
// RUN:   -fsyntax-only -verify %s

void test_too_many_args() {
  DebugBreak(1);
  // expected-error@-1 {{no matching function for call to 'DebugBreak'}}
  // expected-note@hlsl/hlsl_debugging.h:* {{candidate function not viable: requires 0 arguments, but 1 was provided}}

  hlsl::DebugBreak(1);
  // expected-error@-1 {{too many arguments to function call, expected 0, have 1}}
  // expected-note@hlsl/hlsl_debugging.h:* {{'DebugBreak' declared here}}
}
