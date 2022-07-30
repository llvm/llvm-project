// RUN: cat %s | clang-repl -Xcc -Xclang -Xcc -load -Xcc -Xclang \
// RUN:            -Xcc %llvmshlibdir/PrintFunctionNames%pluginext -Xcc -Xclang\
// RUN:            -Xcc -add-plugin -Xcc -Xclang -Xcc print-fns 2>&1 | FileCheck %s
// REQUIRES: plugins, examples

int i = 10;
extern "C" int printf(const char*,...);
auto r1 = printf("i = %d\n", i);
%quit

// CHECK: top-level-decl: "i"
// CHECK-NEXT: top-level-decl: "r1"
// CHECK-NEXT: i = 10
