// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -ast-dump \
// RUN:  -disable-llvm-passes -o - %s | FileCheck %s

// This test ensures that the sample root signature is parsed without error and
// the Attr AST Node is created succesfully. If an invalid root signature was
// passed in then we would exit out of Sema before the Attr is created.

#define SampleRS \
  "DescriptorTable( " \
  "  CBV(), " \
  "  SRV(), " \
  "  UAV()" \
  "), " \
  "DescriptorTable(Sampler())"

// CHECK:      HLSLRootSignatureAttr
// CHECK-SAME: "DescriptorTable(
// CHECK-SAME:   CBV(),
// CHECK-SAME:   SRV(),
// CHECK-SAME:   UAV()
// CHECK-SAME: ),
// CHECK-SAME: DescriptorTable(Sampler())"
[RootSignature(SampleRS)]
void main() {}
