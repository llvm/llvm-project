// Test that xnack and sramecc module flags are emitted based on -m flags
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx90a \
// RUN:   -mxnack -mno-sramecc \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefixes=XNACK-ON,SRAMECC-OFF

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx90a \
// RUN:   -mno-xnack -msramecc \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefixes=XNACK-OFF,SRAMECC-ON

// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx90a \
// RUN:   -emit-llvm -o - %s | FileCheck %s --check-prefix=NO-FLAGS

// XNACK-ON-DAG: !{i32 1, !"amdgpu.xnack", i32 1}
// XNACK-OFF-DAG: !{i32 1, !"amdgpu.xnack", i32 0}
// SRAMECC-ON-DAG: !{i32 1, !"amdgpu.sramecc", i32 1}
// SRAMECC-OFF-DAG: !{i32 1, !"amdgpu.sramecc", i32 0}

// When no explicit xnack/sramecc feature is set, no module flags are emitted
// NO-FLAGS-NOT: !"amdgpu.xnack"
// NO-FLAGS-NOT: !"amdgpu.sramecc"

__attribute__((device)) void test() {}
