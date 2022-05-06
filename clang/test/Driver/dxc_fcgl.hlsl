// RUN: %clang_dxc -fcgl foo.hlsl -### %s 2>&1 | FileCheck %s

// Make sure fcgl option flag which translated into "-S" "-emit-llvm" "-disable-llvm-passes".
// CHECK:"-S" "-emit-llvm" "-disable-llvm-passes"

