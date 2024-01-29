// RUN: clang-repl -Xcc -E
// RUN: clang-repl -Xcc -emit-llvm 
// RUN: clang-repl -oop-executor -Xcc -E
// expected-no-diagnostics
