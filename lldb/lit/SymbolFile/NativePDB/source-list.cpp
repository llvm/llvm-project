// clang-format off
// REQUIRES: lld

// Test that we can set display source of functions.
// RUN: %clang_cl /Z7 /GS- /GR- /c /Fo%t.obj -- %s 
// RUN: lld-link /DEBUG /nodefaultlib /entry:main /OUT:%t.exe /PDB:%t.pdb -- %t.obj
// RUN: env LLDB_USE_NATIVE_PDB_READER=1 %lldb -f %t.exe -s \
// RUN:     %p/Inputs/source-list.lldbinit | FileCheck %s



// Some context lines before
// the function.


int main(int argc, char **argv) {
  // Here are some comments.
  // That we should print when listing source.
  return 0;
}

// Some context lines after
// the function.

// check lines go at the end so that line numbers stay stable when
// changing this file.

// CHECK: (lldb) source list -n main
// CHECK: File: {{.*}}source-list.cpp
// CHECK:    11
// CHECK:    12    // Some context lines before
// CHECK:    13   // the function.
// CHECK:    14
// CHECK:    15
// CHECK:    16   int main(int argc, char **argv) {
// CHECK:    17     // Here are some comments.
// CHECK:    18     // That we should print when listing source.
// CHECK:    19     return 0;
// CHECK:    20   }
// CHECK:    21
// CHECK:    22   // Some context lines after
// CHECK:    23   // the function.
// CHECK:    24
