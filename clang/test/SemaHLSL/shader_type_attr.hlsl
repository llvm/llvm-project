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
[shader("pixel")]
// expected-note@+1 {{conflicting attribute is here}}
[shader("vertex")]
int doubledUp() {
  return 1;
}

// expected-note@+1 {{conflicting attribute is here}}
[shader("vertex")]
int forwardDecl();

// expected-error@+1 {{'shader' attribute parameters do not match the previous declaration}}
[shader("compute")][numthreads(8,1,1)]
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
// expected-warning@+1 {{'shader' attribute argument not supported: library}}
[shader("library")]
#endif // END of FAIL

// CHECK:HLSLShaderAttr 0x{{[0-9a-fA-F]+}} <line:62:2, col:18> Compute
// CHECK:HLSLNumThreadsAttr 0x{{[0-9a-fA-F]+}} <col:21, col:37> 8 1 1
[shader("compute")][numthreads(8,1,1)]
int entry() {
  return 1;
}

// Because these two attributes match, they should both appear in the AST
[shader("compute")][numthreads(8,1,1)]
// CHECK:HLSLShaderAttr 0x{{[0-9a-fA-F]+}} <line:68:2, col:18> Compute
// CHECK:HLSLNumThreadsAttr 0x{{[0-9a-fA-F]+}} <col:21, col:37> 8 1 1
int secondFn();

[shader("compute")][numthreads(8,1,1)]
// CHECK:HLSLShaderAttr 0x{{[0-9a-fA-F]+}} <line:73:2, col:18> Compute
// CHECK:HLSLNumThreadsAttr 0x{{[0-9a-fA-F]+}} <col:21, col:37> 8 1 1
int secondFn() {
  return 1;
}
