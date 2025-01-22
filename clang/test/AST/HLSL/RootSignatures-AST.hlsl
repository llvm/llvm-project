// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -ast-dump \
// RUN:  -disable-llvm-passes -o - %s | FileCheck %s

// This test ensures that the sample root signature is parsed without error and
// the Attr AST Node is created succesfully. If an invalid root signature was
// passed in then we would exit out of Sema before the Attr is created.

#define SampleRS \
  "DescriptorTable( " \
  "  CBV(b1), " \
  "  SRV(t1, numDescriptors = 8, " \
  "          flags = DESCRIPTORS_VOLATILE), " \
  "  UAV(u1, numDescriptors = 0, " \
  "          flags = DESCRIPTORS_VOLATILE) " \
  "), " \
  "DescriptorTable(Sampler(s0, numDescriptors = 4, space = 1))"

// CHECK:      HLSLRootSignatureAttr 0x{{[0-9A-Fa-f]+}} <line:{{[0-9]+}}:{{[0-9]+}}, col:{{[0-9]+}}>
// CHECK-SAME: "DescriptorTable(
// CHECK-SAME:   CBV(b1),
// CHECK-SAME:   SRV(t1, numDescriptors = 8,
// CHECK-SAME:           flags = DESCRIPTORS_VOLATILE),
// CHECK-SAME:   UAV(u1, numDescriptors = 0,
// CHECK-SAME:           flags = DESCRIPTORS_VOLATILE)
// CHECK-SAME: ),
// CHECK-SAME: DescriptorTable(Sampler(s0, numDescriptors = 4, space = 1))"
[RootSignature(SampleRS)]
void main() {}
