// RUN: not %clang -no-integrated-as %s 2>&1 | FileCheck %s
// CHECK: error: unknown type name 'invalid'
// CHECK-NOT: clang: error: assembler command failed
// CHECK-NOT: clang: error: linker command failed
// XFAIL: windows-msvc, ps5
invalid C code!
