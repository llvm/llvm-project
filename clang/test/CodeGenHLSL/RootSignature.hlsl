// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -emit-llvm -o - %s | FileCheck %s

// CHECK-DAG: ![[#EMPTY:]] = !{}
[shader("compute"), RootSignature("")]
[numthreads(1,1,1)]
void FirstEntry() {}

// CHECK-DAG: ![[#TABLE:]] = !{!"DescriptorTable"}
// CHECK-DAG: ![[#SECOND_RS:]] = !{![[#TABLE]]}

#define SampleDescriptorTable \
  "DescriptorTable( " \
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
