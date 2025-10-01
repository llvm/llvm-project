// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -ast-dump \
// RUN:  -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-V1_1
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -ast-dump \
// RUN:  -fdx-rootsignature-version=rootsig_1_0 \
// RUN:  -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-V1_0
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -ast-dump \
// RUN:  -fdx-rootsignature-version=rootsig_1_1 \
// RUN:  -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CHECK-V1_1

// This test ensures that the sample root signature is parsed without error and
// the Attr AST Node is created succesfully. If an invalid root signature was
// passed in then we would exit out of Sema before the Attr is created.

#define SampleRS "RootFlags( ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT | " \
                             "DENY_VERTEX_SHADER_ROOT_ACCESS), " \
                 "CBV(b0, space = 1, flags = DATA_VOLATILE), " \
                 "SRV(t0), " \
                 "UAV(u0), " \
                 "DescriptorTable( CBV(b1), " \
                                   "SRV(t1, numDescriptors = 8, " \
                                   "        flags = DATA_VOLATILE | DESCRIPTORS_VOLATILE), " \
                                   "UAV(u1, numDescriptors = unbounded, " \
                                   "        flags = DATA_VOLATILE | DESCRIPTORS_VOLATILE)), " \
                 "DescriptorTable(Sampler(s0, space=1, numDescriptors = 4)), " \
                 "RootConstants(num32BitConstants=3, b10), " \
                 "StaticSampler(s1)," \
                 "StaticSampler(s2, " \
                                 "addressU = TEXTURE_ADDRESS_CLAMP, " \
                                 "filter = FILTER_MIN_MAG_MIP_LINEAR )"

// CHECK: -HLSLRootSignatureDecl 0x{{.*}} {{.*}} implicit [[SAMPLE_RS_DECL:__hlsl_rootsig_decl_\d*]]
// CHECK-V1_0: version: 1.0,
// CHECK-V1_1: version: 1.1,
// CHECK-SAME: RootElements{
// CHECK-SAME: RootFlags(AllowInputAssemblerInputLayout | DenyVertexShaderRootAccess),
// CHECK-SAME: RootCBV(b0,
// CHECK-SAME:   space = 1, visibility = All, flags = DataVolatile
// CHECK-SAME: ),
// CHECK-SAME: RootSRV(t0,
// CHECK-SAME:   space = 0, visibility = All,
// CHECK-V1_0-SAME: flags = DataVolatile
// CHECK-V1_1-SAME: flags = DataStaticWhileSetAtExecute
// CHECK-SAME: ),
// CHECK-SAME: RootUAV(
// CHECK-SAME:   u0, space = 0, visibility = All, flags = DataVolatile
// CHECK-SAME: ),
// CHECK-SAME: CBV(
// CHECK-SAME:   b1, numDescriptors = 1, space = 0, offset = DescriptorTableOffsetAppend,
// CHECK-V1_0-SAME: flags = DescriptorsVolatile | DataVolatile
// CHECK-V1_1-SAME: flags = DataStaticWhileSetAtExecute
// CHECK-SAME: ),
// CHECK-SAME: SRV(
// CHECK-SAME:   t1, numDescriptors = 8, space = 0, offset = DescriptorTableOffsetAppend, flags = DescriptorsVolatile | DataVolatile
// CHECK-SAME: ),
// CHECK-SAME: UAV(
// CHECK-SAME:   u1, numDescriptors = unbounded, space = 0, offset = DescriptorTableOffsetAppend, flags = DescriptorsVolatile | DataVolatile
// CHECK-SAME: ),
// CHECK-SAME: DescriptorTable(
// CHECK-SAME:   numClauses = 3, visibility = All
// CHECK-SAME: ),
// CHECK-SAME: Sampler(
// CHECK-SAME:   s0, numDescriptors = 4, space = 1, offset = DescriptorTableOffsetAppend,
// CHECK-V1_0-SAME:  flags = DescriptorsVolatile
// CHECK-V1_1-SAME:  flags = None
// CHECK-SAME: ),
// CHECK-SAME: DescriptorTable(
// CHECK-SAME:   numClauses = 1, visibility = All
// CHECK-SAME: ),
// CHECK-SAME: RootConstants(
// CHECK-SAME:   num32BitConstants = 3, b10, space = 0, visibility = All
// CHECK-SAME: ),
// CHECK-SAME: StaticSampler(
// CHECK-SAME:   s1, filter = Anisotropic, addressU = Wrap, addressV = Wrap, addressW = Wrap,
// CHECK-SAME:   mipLODBias = 0.000000e+00, maxAnisotropy = 16, comparisonFunc = LessEqual,
// CHECK-SAME:   borderColor = OpaqueWhite, minLOD = 0.000000e+00, maxLOD = 3.402823e+38, space = 0, visibility = All
// CHECK-SAME: )}

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
   "RootFlags( ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT | " \
               "DENY_VERTEX_SHADER_ROOT_ACCESS), " \
   "CBV(b0, space = 1, flags = DATA_VOLATILE), " \
   "SRV(t0), " \
   "UAV(u0), " \
   "DescriptorTable( CBV(b1), " \
   "  SRV(t1, numDescriptors = 8, " \
   "          flags = DATA_VOLATILE | DESCRIPTORS_VOLATILE), " \
   "  UAV(u1, numDescriptors = unbounded, " \
   "          flags = DATA_VOLATILE | DESCRIPTORS_VOLATILE)), " \
   "DescriptorTable(Sampler(s0, space=1, numDescriptors = 4)), " \
   "RootConstants(num32BitConstants=3, b10), " \
   "StaticSampler(s1)," \
   "StaticSampler(s2, " \
                   "addressU = TEXTURE_ADDRESS_CLAMP, " \
                   "filter = FILTER_MIN_MAG_MIP_LINEAR )"

// CHECK: -RootSignatureAttr 0x{{.*}} {{.*}} [[SAMPLE_RS_DECL]]
[RootSignature(SampleSameRS)]
void same_rs_string_main() {}

#define SampleDifferentRS \
  "DescriptorTable(Sampler(s0, numDescriptors = 4, space = 1))"

// Ensure that when we define a different type root signature that it creates
// a separate decl and identifier to reference

// CHECK: -HLSLRootSignatureDecl 0x{{.*}} {{.*}} implicit [[DIFF_RS_DECL:__hlsl_rootsig_decl_\d*]]
// CHECK-V1_0: version: 1.0,
// CHECK-V1_1: version: 1.1,
// CHECK-SAME: RootElements{
// CHECK-SAME:   Sampler(s0, numDescriptors = 4, space = 1, offset = DescriptorTableOffsetAppend,
// CHECK-V1_0-SAME:  flags = DescriptorsVolatile
// CHECK-V1_1-SAME:  flags = None
// CHECK-SAME: ),
// CHECK-SAME:   DescriptorTable(numClauses = 1, visibility = All)
// CHECK-SAME: }

// CHECK: -RootSignatureAttr 0x{{.*}} {{.*}} [[DIFF_RS_DECL]]
[RootSignature(SampleDifferentRS)]
void different_rs_string_main() {}
