// RUN: %clang_dxc -T cs_6_0 -fcgl %s | FileCheck %s --check-prefix=CHECK-V1_1

// RUN: %clang_dxc -T cs_6_0 -fcgl -force-rootsig-ver rootsig_1_0 %s | FileCheck %s --check-prefix=CHECK-V1_0
// RUN: %clang_dxc -T cs_6_0 -fcgl -force-rootsig-ver rootsig_1_1 %s | FileCheck %s --check-prefix=CHECK-V1_1

// Test to demonstrate that we can specify the root-signature versions

// CHECK: !dx.rootsignatures = !{![[#EMPTY_ENTRY:]]}
// CHECK: ![[#EMPTY_ENTRY]] = !{ptr @EmptyEntry, ![[#EMPTY:]],
// CHECK-V1_0: i32 1}
// CHECK-V1_1: i32 2}
// CHECK: ![[#EMPTY]] = !{}

[shader("compute"), RootSignature("")]
[numthreads(1,1,1)]
void EmptyEntry() {}
