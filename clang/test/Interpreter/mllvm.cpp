// LLVM options reach the JIT's TargetMachine for every input.
// RUN: cat %s | clang-repl -Xcc -mllvm -Xcc -print-after-isel 2>&1 | FileCheck %s
// RUN: cat %s | clang-repl -print-after-isel 2>&1 | FileCheck %s

extern "C" int f1() { return 1; }
int v1 = f1();
// CHECK: # Machine code for function f1:

extern "C" int f2() { return 2; }
int v2 = f2();
// CHECK: # Machine code for function f2:
