// RUN: %clang_dxc -fcgl -T lib_6_7 %s -### %s 2>&1 | FileCheck %s

// Make sure fcgl option flag which translated into "-emit-llvm" "-disable-llvm-passes".
// CHECK: "-emit-llvm"
// CHECK-SAME: "-disable-llvm-passes"
