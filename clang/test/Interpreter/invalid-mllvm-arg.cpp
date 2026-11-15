// RUN: not clang-repl --Xcc=-mllvm --Xcc=-helium 2>&1 | FileCheck %s

// An invalid -mllvm argument is reported, and clang-repl fails to start.
// CHECK: clang (LLVM option parsing): Unknown command line argument '-helium'
// CHECK: error: invalid argument in '-mllvm'
// CHECK: clang-repl: Failed to parse -mllvm arguments
