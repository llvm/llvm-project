// RUN: %clang_dxc -I test  -### %s 2>&1 | FileCheck %s

// Make sure -I send to cc1.
// CHECK:"-I" "test"
