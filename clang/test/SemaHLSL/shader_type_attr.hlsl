// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -ast-dump -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -ast-dump -o - %s -DFAIL -verify

// FileCheck test make sure HLSLShaderAttr is generated in AST.
// verify test make sure validation on shader type attribute works as expected.

#ifdef FAIL

// expected-warning@+1 {{'shader' attribute only applies to global functions}}
[shader("compute")]
struct Fido {
  // expected-warning@+1 {{'shader' attribute only applies to global functions}}
  [shader("pixel")]
  void wag() {}
  // expected-warning@+1 {{'shader' attribute only applies to global functions}}
  [shader("vertex")]
  static void oops() {}
};

// expected-warning@+1 {{'shader' attribute only applies to global functions}}
[shader("vertex")]
static void oops() {}

namespace spec {
// expected-warning@+1 {{'shader' attribute only applies to global functions}}
[shader("vertex")]
static void oops() {}
} // namespace spec

// expected-error@+1 {{'shader' attribute parameters do not match the previous declaration}}
[shader("compute")]
// expected-note@+1 {{conflicting attribute is here}}
[shader("vertex")]
int doubledUp() {
  return 1;
}

// expected-note@+1 {{conflicting attribute is here}}
[shader("vertex")]
int forwardDecl();

// expected-error@+1 {{'shader' attribute parameters do not match the previous declaration}}
[shader("compute")]
int forwardDecl() {
  return 1;
}

// expected-error@+1 {{'shader' attribute takes one argument}}
[shader()]
// expected-error@+1 {{'shader' attribute takes one argument}}
[shader(1, 2)]
// expected-error@+1 {{'shader' attribute requires a string}}
[shader(1)]
// expected-warning@+1 {{'shader' attribute argument not supported: cs}}
[shader("cs")]

#endif // END of FAIL

// CHECK:HLSLShaderAttr 0x{{[0-9a-fA-F]+}} <line:60:2, col:18> Compute
[shader("compute")]
int entry() {
  return 1;
}

// Because these two attributes match, they should both appear in the AST
[shader("compute")]
// CHECK:HLSLShaderAttr 0x{{[0-9a-fA-F]+}} <line:66:2, col:18> Compute
int secondFn();

[shader("compute")]
// CHECK:HLSLShaderAttr 0x{{[0-9a-fA-F]+}} <line:70:2, col:18> Compute
int secondFn() {
  return 1;
}
