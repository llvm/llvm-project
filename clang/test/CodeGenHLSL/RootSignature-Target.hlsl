// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-rootsignature \
// RUN: -hlsl-entry EntryRS -emit-llvm -o - %s | FileCheck %s

// CHECK: !dx.rootsignatures = !{![[#ENTRY:]]}
// CHECK: ![[#ENTRY]] = !{null, ![[#ENTRY_RS:]], i32 2}
// CHECK: ![[#ENTRY_RS]] = !{![[#ROOT_CBV:]]}
// CHECK: ![[#ROOT_CBV]] = !{!"RootCBV", i32 0, i32 0, i32 0, i32 4}

#define EntryRS "CBV(b0)"
