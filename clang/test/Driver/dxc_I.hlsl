// RUN: %clang_dxc -I test -Tlib_6_3  -### %s 2>&1 | FileCheck %s

// Make sure -I send to cc1.
// CHECK:"-I" "test"
