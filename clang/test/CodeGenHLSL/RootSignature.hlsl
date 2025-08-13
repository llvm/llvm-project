// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -emit-llvm -o - %s | FileCheck %s

// CHECK: !dx.rootsignatures = !{![[#EMPTY_ENTRY:]], ![[#DT_ENTRY:]],
// CHECK-SAME: ![[#RF_ENTRY:]], ![[#RC_ENTRY:]], ![[#RD_ENTRY:]], ![[#SS_ENTRY:]]}

// CHECK: ![[#EMPTY_ENTRY]] = !{ptr @EmptyEntry, ![[#EMPTY:]], i32 2}
// CHECK: ![[#EMPTY]] = !{}

[shader("compute"), RootSignature("")]
[numthreads(1,1,1)]
void EmptyEntry() {}

// CHECK: ![[#DT_ENTRY]] = !{ptr @DescriptorTableEntry, ![[#DT_RS:]], i32 2}
// CHECK: ![[#DT_RS]] = !{![[#TABLE:]]}
// CHECK: ![[#TABLE]] = !{!"DescriptorTable", i32 0, ![[#CBV:]], ![[#SRV:]]}
// CHECK: ![[#CBV]] = !{!"CBV", i32 1, i32 0, i32 0, i32 -1, i32 4}
// CHECK: ![[#SRV]] = !{!"SRV", i32 4, i32 42, i32 3, i32 32, i32 0}

#define SampleDescriptorTable \
  "DescriptorTable( " \
  "  CBV(b0), " \
  "  SRV(t42, space = 3, offset = 32, numDescriptors = 4, flags = 0) " \
  ")"
[shader("compute"), RootSignature(SampleDescriptorTable)]
[numthreads(1,1,1)]
void DescriptorTableEntry() {}

// CHECK: ![[#RF_ENTRY]] = !{ptr @RootFlagsEntry, ![[#RF_RS:]], i32 2}
// CHECK: ![[#RF_RS]] = !{![[#ROOT_FLAGS:]]}
// CHECK: ![[#ROOT_FLAGS]] = !{!"RootFlags", i32 2114}

#define SampleRootFlags \
  "RootFlags( " \
  " Deny_Vertex_Shader_Root_Access | Allow_Stream_Output | " \
  " sampler_heap_directly_indexed " \
  ")"
[shader("compute"), RootSignature(SampleRootFlags)]
[numthreads(1,1,1)]
void RootFlagsEntry() {}

// CHECK: ![[#RC_ENTRY]] = !{ptr @RootConstantsEntry, ![[#RC_RS:]], i32 2}
// CHECK: ![[#RC_RS]] = !{![[#ROOT_CONSTANTS:]]}
// CHECK: ![[#ROOT_CONSTANTS]] = !{!"RootConstants", i32 5, i32 1, i32 2, i32 1}

#define SampleRootConstants \
  "RootConstants(" \
  " space = 2, " \
  " visibility = Shader_Visibility_Pixel, " \
  " b1, num32BitConstants = 1 " \
  ")"
[shader("compute"), RootSignature(SampleRootConstants)]
[numthreads(1,1,1)]
void RootConstantsEntry() {}

// CHECK: ![[#RD_ENTRY]] = !{ptr @RootDescriptorsEntry, ![[#RD_RS:]], i32 2}
// CHECK: ![[#RD_RS]] = !{![[#ROOT_CBV:]], ![[#ROOT_UAV:]], ![[#ROOT_SRV:]]}
// CHECK: ![[#ROOT_CBV]] = !{!"RootCBV", i32 0, i32 0, i32 0, i32 4}
// CHECK: ![[#ROOT_UAV]] = !{!"RootUAV", i32 0, i32 42, i32 3, i32 2}
// CHECK: ![[#ROOT_SRV]] = !{!"RootSRV", i32 4, i32 0, i32 0, i32 2}

#define SampleRootDescriptors \
  "CBV(b0), " \
  "UAV(space = 3, u42), " \
  "SRV(t0, visibility = Shader_Visibility_Geometry, flags = Data_Volatile)"
[shader("compute"), RootSignature(SampleRootDescriptors)]
[numthreads(1,1,1)]
void RootDescriptorsEntry() {}

// CHECK: ![[#SS_ENTRY]] = !{ptr @StaticSamplerEntry, ![[#SS_RS:]], i32 2}
// CHECK: ![[#SS_RS]] = !{![[#STATIC_SAMPLER:]]}

// checking filter = 0x4
// CHECK: ![[#STATIC_SAMPLER]] = !{!"StaticSampler", i32 4,

// checking texture address[U|V|W]
// CHECK-SAME: i32 2, i32 3, i32 5,

// checking mipLODBias, maxAnisotropy, comparisonFunc, borderColor
// note: the hex value is the float bit representation of 12.45
// CHECK-SAME: float 0x4028E66660000000, i32 9, i32 3, i32 2,

// checking minLOD, maxLOD
// CHECK-SAME: float -1.280000e+02, float 1.280000e+02,

// checking register, space and visibility
// CHECK-SAME: i32 42, i32 0, i32 0}

#define SampleStaticSampler \
  "StaticSampler(s42, " \
  " filter = FILTER_MIN_POINT_MAG_LINEAR_MIP_POINT, " \
  " addressU = TEXTURE_ADDRESS_MIRROR, " \
  " addressV = TEXTURE_ADDRESS_CLAMP, " \
  " addressW = TEXTURE_ADDRESS_MIRRORONCE, " \
  " mipLODBias = 12.45f, maxAnisotropy = 9, " \
  " comparisonFunc = COMPARISON_EQUAL, " \
  " borderColor = STATIC_BORDER_COLOR_OPAQUE_WHITE, " \
  " minLOD = -128.f, maxLOD = 128.f, " \
  " space = 0, visibility = SHADER_VISIBILITY_ALL, " \
  ")"
[shader("compute"), RootSignature(SampleStaticSampler)]
[numthreads(1,1,1)]
void StaticSamplerEntry() {}

// Sanity test to ensure no root is added for this function as there is only
// two entries in !dx.roosignatures
[shader("compute")]
[numthreads(1,1,1)]
void NoRSEntry() {}
