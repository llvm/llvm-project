// RUN: not %clang_dxc -Efoo -Tlib_6_7 foo.hlsl -### %s 2>&1 | FileCheck %s

// Make sure E option flag which translated into "-hlsl-entry".
// CHECK:"-hlsl-entry" "foo"
