// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -emit-llvm -o - %s | FileCheck %s

// CHECK: !dx.rootsignatures = !{![[#EMPTY_ENTRY:]], ![[#DT_ENTRY:]]}

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

// Sanity test to ensure no root is added for this function as there is only
// two entries in !dx.roosignatures
[shader("compute")]
[numthreads(1,1,1)]
void NoRSEntry() {}
