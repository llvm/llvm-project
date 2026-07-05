// No errors. Otherwise nothing observable.
// RUN: %clang_dxc -fvk-use-dx-layout -spirv -Tlib_6_7 -### %s

// RUN: not %clang_dxc -fvk-use-scalar-layout -spirv -Tlib_6_7 -### %s 2>&1 | FileCheck %s -check-prefix=SCALAR
// SCALAR: error: the clang compiler does not support '-fvk-use-scalar-layout'

// RUN: not %clang_dxc -fvk-use-gl-layout -spirv -Tlib_6_7 -### %s 2>&1 | FileCheck %s -check-prefix=GL
// GL: error: the clang compiler does not support '-fvk-use-gl-layout' 
