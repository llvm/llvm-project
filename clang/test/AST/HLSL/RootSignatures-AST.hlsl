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

// CHECK: -HLSLRootSignatureDecl 0x{{.*}} {{.*}} implicit [[SAMPLE_RS_DECL:__hlsl_rootsig_decl_\d*]]
// CHECK-SAME: RootElements{
// CHECK-SAME:   CBV(b1, numDescriptors = 1, space = 0,
// CHECK-SAME:     offset = DescriptorTableOffsetAppend, flags = DataStaticWhileSetAtExecute),
// CHECK-SAME:   SRV(t1, numDescriptors = 8, space = 0,
// CHECK-SAME:     offset = DescriptorTableOffsetAppend, flags = DescriptorsVolatile),
// CHECK-SAME:   UAV(u1, numDescriptors = 0, space = 0,
// CHECK-SAME:     offset = DescriptorTableOffsetAppend, flags = DescriptorsVolatile),
// CHECK-SAME:   DescriptorTable(numClauses = 3, visibility = All),
// CHECK-SAME:   Sampler(s0, numDescriptors = 4, space = 1,
// CHECK-SAME:     offset = DescriptorTableOffsetAppend, flags = None),
// CHECK-SAME:   DescriptorTable(numClauses = 1, visibility = All)
// CHECK-SAME: }

// CHECK: -RootSignatureAttr 0x{{.*}} {{.*}} [[SAMPLE_RS_DECL]]
[RootSignature(SampleRS)]
void rs_main() {}

// Ensure that if multiple root signatures are specified at different entry
// points that we point to the correct root signature

// CHECK: -RootSignatureAttr 0x{{.*}} {{.*}} [[SAMPLE_RS_DECL]]
[RootSignature(SampleRS)]
void same_rs_main() {}

// Define the same root signature to ensure that the entry point will still
// link to the same root signature declaration

#define SampleSameRS \
  "DescriptorTable( " \
  "  CBV(b1), " \
  "  SRV(t1, numDescriptors = 8, " \
  "          flags = DESCRIPTORS_VOLATILE), " \
  "  UAV(u1, numDescriptors = 0, " \
  "          flags = DESCRIPTORS_VOLATILE) " \
  "), " \
  "DescriptorTable(Sampler(s0, numDescriptors = 4, space = 1))"

// CHECK: -RootSignatureAttr 0x{{.*}} {{.*}} [[SAMPLE_RS_DECL]]
[RootSignature(SampleSameRS)]
void same_rs_string_main() {}

#define SampleDifferentRS \
  "DescriptorTable(Sampler(s0, numDescriptors = 4, space = 1))"

// Ensure that when we define a different type root signature that it creates
// a seperate decl and identifier to reference

// CHECK: -HLSLRootSignatureDecl 0x{{.*}} {{.*}} implicit [[DIFF_RS_DECL:__hlsl_rootsig_decl_\d*]]
// CHECK-SAME: RootElements{
// CHECK-SAME:   Sampler(s0, numDescriptors = 4, space = 1,
// CHECK-SAME:     offset = DescriptorTableOffsetAppend, flags = None),
// CHECK-SAME:   DescriptorTable(numClauses = 1, visibility = All)
// CHECK-SAME: }

// CHECK: -RootSignatureAttr 0x{{.*}} {{.*}} [[DIFF_RS_DECL]]
[RootSignature(SampleDifferentRS)]
void different_rs_string_main() {}
