// RUN: %clang_dxc -fcgl -T lib_6_7 foo.hlsl -### %s 2>&1 | FileCheck %s

// Make sure fcgl option flag which translated into "-S" "-emit-llvm" "-disable-llvm-passes".
// CHECK:"-S" "-emit-llvm" "-disable-llvm-passes"

