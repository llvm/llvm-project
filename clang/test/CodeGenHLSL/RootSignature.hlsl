// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -emit-llvm -o - %s | FileCheck %s

// CHECK: !dx.rootsignatures = !{![[#EMPTY_ENTRY:]], ![[#DT_ENTRY:]],
// CHECK-SAME: ![[#RF_ENTRY:]], ![[#RC_ENTRY:]]}

// CHECK: ![[#EMPTY_ENTRY]] = !{ptr @EmptyEntry, ![[#EMPTY:]]}
// CHECK: ![[#EMPTY]] = !{}

[shader("compute"), RootSignature("")]
[numthreads(1,1,1)]
void EmptyEntry() {}

// CHECK: ![[#DT_ENTRY]] = !{ptr @DescriptorTableEntry, ![[#DT_RS:]]}
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

// CHECK: ![[#RF_ENTRY]] = !{ptr @RootFlagsEntry, ![[#RF_RS:]]}
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

// CHECK: ![[#RC_ENTRY]] = !{ptr @RootConstantsEntry, ![[#RC_RS:]]}
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

// Sanity test to ensure no root is added for this function as there is only
// two entries in !dx.roosignatures
[shader("compute")]
[numthreads(1,1,1)]
void NoRSEntry() {}
