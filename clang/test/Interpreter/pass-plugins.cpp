// FIXME: Add FileCheck back once plugin parameters work without -load
//
// RUN: cat %s | clang-repl -Xcc -fpass-plugin=%llvmshlibdir/Bye%pluginext \
// RUN:                     -Xcc -Xclang -Xcc -load -Xcc -Xclang -Xcc %llvmshlibdir/Bye%pluginext \
// RUN:                     -Xcc -mllvm -Xcc -wave-goodbye
// REQUIRES: plugins, llvm-examples

int i = 10;
%quit

// CHECK: Bye
