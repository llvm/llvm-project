// RUN: cat %s | clang-repl -Xcc -fpass-plugin=%plugindir/pypass-plugin%pluginext \
// RUN:                     -Xcc -load=%plugindir/pypass-plugin%pluginext \
// RUN:                     -Xcc -Xclang -Xcc -mllvm -Xcc -wave-goodbye | FileCheck %s
// REQUIRES: plugins, llvm-examples

int i = 10;
%quit

// CHECK: Bye
