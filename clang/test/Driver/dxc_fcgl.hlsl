// RUN: not %clang_dxc -fcgl -T lib_6_7 foo.hlsl -### %s 2>&1 | FileCheck %s
// RUN: %clang_dxc -fcgl -T lib_6_7 %s -Xclang -verify

// Make sure fcgl option flag which translated into "-emit-llvm" "-disable-llvm-passes".
// CHECK: "-emit-llvm"
// CHECK-SAME: "-disable-llvm-passes"

// Make sure fcgl option not generate any diagnostics.
// expected-no-diagnostics
