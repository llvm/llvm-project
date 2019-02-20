// clang-format off
// REQUIRES: lld

// Test that we can set simple breakpoints using PDB on any platform.
// RUN: %build --compiler=clang-cl --nodefaultlib -o %t.exe -- %s
// RUN: env LLDB_USE_NATIVE_PDB_READER=1 %lldb -f %t.exe -s \
// RUN:     %p/Inputs/break-by-line.lldbinit | FileCheck %s

// This is a separate test from break-by-function.cpp because this test is
// sensitive to edits in the source file.

namespace NS {
  int NamespaceFn(int X) {
    return X + 42;
  }
}

int main(int argc, char **argv) {
  return NS::NamespaceFn(argc);
}


// CHECK:      (lldb) target create "{{.*}}break-by-line.cpp.tmp.exe"
// CHECK:      Current executable set to '{{.*}}break-by-line.cpp.tmp.exe'
// CHECK:      (lldb) break set -f break-by-line.cpp -l 14
// CHECK:      Breakpoint 1: where = break-by-line.cpp.tmp.exe`NS::NamespaceFn + {{[0-9]+}} at break-by-line.cpp:14
