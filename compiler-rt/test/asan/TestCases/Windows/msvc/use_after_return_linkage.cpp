// Make sure LIBCMT doesn't accidentally get added to the list of DEFAULTLIB
// directives.

// REQUIRES: asan-dynamic-runtime
// RUN: %clang_cl_asan -c %s -Fo%t.obj
// RUN: llvm-readobj --coff-directives %t.obj | FileCheck %s
// RUN: %clang_cl_asan -LD %t.obj -Fe%t.dll
// CHECK-NOT: {{[Ll][Ii][Bb][Cc][Mm][Tt]}}
// CHECK: /DEFAULTLIB:msvcrt.lib
// CHECK-NOT: {{[Ll][Ii][Bb][Cc][Mm][Tt]}}

void foo(int *p) { *p = 42; }

__declspec(dllexport) void bar() {
  int x;
  foo(&x);
}
