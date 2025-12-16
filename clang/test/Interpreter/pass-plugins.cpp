// RUN: cat %s | clang-repl -Xcc -fpass-plugin=%llvmshlibdir/Bye%pluginext \
// RUN:                     -Xcc -Xclang -Xcc -load -Xcc -Xclang -Xcc %llvmshlibdir/Bye%pluginext \
// RUN:                     -Xcc -mllvm -Xcc -wave-goodbye | FileCheck %s
// REQUIRES: plugins, llvm-examples

int i = 10;
%quit

// CHECK: Bye
