// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -emit-llvm -o - %s | FileCheck %s

// CHECK-DAG: ![[#EMPTY:]] = !{}
[shader("compute"), RootSignature("")]
[numthreads(1,1,1)]
void FirstEntry() {}

// CHECK-DAG: ![[#CBV:]] = !{!"CBV", i32 1, i32 0, i32 0, i32 -1, i32 4}
// CHECK-DAG: ![[#SRV:]] = !{!"SRV", i32 4, i32 42, i32 3, i32 32, i32 0}
// CHECK-DAG: ![[#TABLE:]] = !{!"DescriptorTable", i32 0, ![[#CBV]], ![[#SRV]]}
// CHECK-DAG: ![[#SECOND_RS:]] = !{![[#TABLE]]}

#define SampleDescriptorTable \
  "DescriptorTable( " \
  "  CBV(b0), " \
  "  SRV(t42, space = 3, offset = 32, numDescriptors = 4, flags = 0) " \
  ")"
[shader("compute"), RootSignature(SampleDescriptorTable)]
[numthreads(1,1,1)]
void SecondEntry() {}

// Sanity test to ensure to root is added for this function
[shader("compute")]
[numthreads(1,1,1)]
void ThirdEntry() {}

// CHECK-DAG: ![[#FIRST_ENTRY:]] = !{ptr @FirstEntry, ![[#EMPTY]]}
// CHECK-DAG: ![[#SECOND_ENTRY:]] = !{ptr @SecondEntry, ![[#SECOND_RS]]}
// CHECK-DAG: !dx.rootsignatures = !{![[#FIRST_ENTRY]], ![[#SECOND_ENTRY]]}
