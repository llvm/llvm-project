// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -emit-llvm -o - %s | FileCheck %s

// CHECK: !dx.rootsignatures = !{![[#FIRST_ENTRY:]], ![[#SECOND_ENTRY:]]}
// CHECK-DAG: ![[#FIRST_ENTRY]] = !{ptr @FirstEntry, ![[#RS:]]}
// CHECK-DAG: ![[#SECOND_ENTRY]] = !{ptr @SecondEntry, ![[#RS:]]}
// CHECK-DAG: ![[#RS]] = !{}

[shader("compute"), RootSignature("")]
[numthreads(1,1,1)]
void FirstEntry() {}

[shader("compute"), RootSignature("DescriptorTable()")]
[numthreads(1,1,1)]
void SecondEntry() {}

// Sanity test to ensure to root is added for this function
[shader("compute")]
[numthreads(1,1,1)]
void ThirdEntry() {}
