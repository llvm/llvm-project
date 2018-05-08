// Check that -dwarf-column-info does not get added to the cc1 line:
// 1) When -gcodeview is present via the clang or clang++ driver
// 2) When /Z7 is present via the cl driver.

// RUN: %clang -### --target=x86_64-windows-msvc -c -g -gcodeview %s 2> %t1
// RUN: FileCheck < %t1 %s
// RUN: %clangxx -### --target=x86_64-windows-msvc -c -g -gcodeview %s 2> %t2
// RUN: FileCheck < %t2 %s
// RUN: %clangxx -### --target=x86_64-windows-gnu -c -g -gcodeview %s 2> %t2
// RUN: FileCheck < %t2 %s
// RUN: %clang_cl -### --target=x86_64-windows-msvc /c /Z7 -- %s 2> %t2
// RUN: FileCheck < %t2 %s

// CHECK: "-cc1"
// CHECK-NOT: "-dwarf-column-info"
