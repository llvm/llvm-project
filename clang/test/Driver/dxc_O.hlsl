// RUN: %clang_dxc -T lib_6_7  foo.hlsl -### %s 2>&1 | FileCheck %s
// RUN: %clang_dxc -T lib_6_7 -Od foo.hlsl -### %s 2>&1 | FileCheck %s --check-prefix=Od
// RUN: %clang_dxc -T lib_6_7 -O0 foo.hlsl -### %s 2>&1 | FileCheck %s --check-prefix=O0
// RUN: %clang_dxc -T lib_6_7 -O1 foo.hlsl -### %s 2>&1 | FileCheck %s --check-prefix=O1
// RUN: %clang_dxc -T lib_6_7 -O2 foo.hlsl -### %s 2>&1 | FileCheck %s --check-prefix=O2
// RUN: %clang_dxc -T lib_6_7 -O3 foo.hlsl -### %s 2>&1 | FileCheck %s --check-prefix=O3

// Make sure default is O3.
// CHECK: "-O3"

// Make sure Od/O0 option flag which translated into "-O0".
// Od: "-O0"
// O0: "-O0"

// Make sure O1/O2/O3 is send to cc1.
// O1: "-O1"
// O2: "-O2"
// O3: "-O3"
