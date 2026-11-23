// RUN: %clang_dxc -T cs_6_0 -fcgl %s | FileCheck %s --check-prefixes=CHECK,REG
// RUN: %clang_dxc -T cs_6_0 -fcgl -rootsig-define EmptyRS %s | FileCheck %s --check-prefixes=CHECK,EMPTY
// RUN: %clang_dxc -T cs_6_0 -fcgl -rootsig-define CmdRS -D CmdRS='"SRV(t0)"' %s | FileCheck %s --check-prefixes=CHECK,CMD

// Equivalent clang checks:
// RUN: %clang -target dxil-unknown-shadermodel6.0-compute -S -emit-llvm -o - %s \
// RUN:  | FileCheck %s --check-prefixes=CHECK,REG

// RUN: %clang -target dxil-unknown-shadermodel6.0-compute -S -emit-llvm -o - %s \
// RUN:  -fdx-rootsignature-define=EmptyRS \
// RUN:  | FileCheck %s --check-prefixes=CHECK,EMPTY

// RUN: %clang -target dxil-unknown-shadermodel6.0-compute -S -emit-llvm -o - %s \
// RUN:  -fdx-rootsignature-define=CmdRS -D CmdRS='"SRV(t0)"' \
// RUN:  | FileCheck %s --check-prefixes=CHECK,CMD

#define EmptyRS ""
#define NotEmptyRS "CBV(b0)"

// CHECK: !dx.rootsignatures = !{![[#ENTRY:]]}
// CHECK: ![[#ENTRY]] = !{ptr @main, ![[#RS:]], i32 2}

// REG: ![[#RS]] = !{![[#CBV:]]}
// REG: ![[#CBV]] = !{!"RootCBV"

// EMPTY: ![[#RS]] = !{}

// CMD: ![[#RS]] = !{![[#SRV:]]}
// CMD: ![[#SRV]] = !{!"RootSRV"

[shader("compute"), RootSignature(NotEmptyRS)]
[numthreads(1,1,1)]
void main() {}
