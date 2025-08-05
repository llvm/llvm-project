// RUN: %clang_cc1 -Qdx-rootsignature-strip -triple dxil-pc-shadermodel6.3-compute -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,FLAG
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-compute -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,NOFLAG

// CHECK: !dx.rootsignatures = !{![[#LOWER_INFO:]], ![[#EMPTY_ENTRY:]]}
// FLAG: ![[#LOWER_INFO]] = !{i1 true}
// NOFLAG: ![[#LOWER_INFO]] = !{i1 false}

// Ensure root signature metadata is still generated in either case
// CHECK: ![[#EMPTY_ENTRY]] = !{ptr @EmptyEntry, ![[#EMPTY:]], i32 2}
// CHECK: ![[#EMPTY]] = !{}

[shader("compute"), RootSignature("")]
[numthreads(1,1,1)]
void EmptyEntry() {}

