// REQUIRES: host-supports-jit
// RUN: cat %s | clang-repl -Xcc -x -Xcc c -Xcc -std=c17 -Xcc -fno-builtin -Xcc -Wno-error=implicit-function-declaration 2>&1 | FileCheck %s
// RUN: cat %s | clang-repl -Xcc -x -Xcc c -Xcc -std=c17 -Xcc -fno-builtin -Xcc -O2 -Xcc -Wno-error=implicit-function-declaration 2>&1 | FileCheck %s

a();

void a() { return; }
// CHECK: error: conflicting types

p();
%undo

void p() {  }
// CHECK-NOT: error: conflicting types

int x = 10;
%quit
