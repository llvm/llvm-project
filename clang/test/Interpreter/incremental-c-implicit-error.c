// REQUIRES: host-supports-jit
// RUN: cat %s | clang-repl -Xcc -x -Xcc c -Xcc -std=c17 -Xcc -fno-builtin 2>&1 | FileCheck %s
// RUN: cat %s | clang-repl -Xcc -x -Xcc c -Xcc -std=c17 -Xcc -fno-builtin -Xcc -O2 2>&1 | FileCheck %s
// see https://github.com/llvm/llvm-project/issues/171440.

a();
// CHECK: error: call to undeclared function 'a'
// CHECK: ISO C99 and later do not support implicit function declarations

void a() { return; }
// CHECK-NOT: error: conflicting types

a();
// CHECK-NOT: Symbols not found


int x = 10;
%quit
