// UNSUPPORTED: system-aix
//
// RUN: clang-repl -Xcc -E
// RUN: clang-repl -Xcc -emit-llvm
// RUN: clang-repl -Xcc -xc
// expected-no-diagnostics
