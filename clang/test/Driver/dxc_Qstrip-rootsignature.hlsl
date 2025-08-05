// RUN: %clang_dxc -Qstrip-rootsignature -T cs_6_3 -HV 202x -Vd -Xclang -emit-llvm %s | FileCheck %s --check-prefixes=CHECK,FLAG
// RUN: %clang_dxc -T cs_6_3 -HV 202x -Vd -Xclang -emit-llvm %s | FileCheck %s --check-prefixes=CHECK,NOFLAG

// Test to demonstrate that we can specify when to strip the root signature
// in its metadata

// CHECK: !dx.rootsignatures = !{![[#LOWER_INFO:]], ![[#EMPTY_ENTRY:]]}
// FLAG: ![[#LOWER_INFO]] = !{i1 true}
// NOFLAG: ![[#LOWER_INFO]] = !{i1 false}

// Ensure root signature metadata is still generated in either case
// CHECK: ![[#EMPTY_ENTRY]] = !{ptr @EmptyEntry, ![[#EMPTY:]], i32 2}
// CHECK: ![[#EMPTY]] = !{}

[shader("compute"), RootSignature("")]
[numthreads(1,1,1)]
void EmptyEntry() {}
