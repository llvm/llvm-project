// Check that these simple command lines for listing LLVM options are supported,
// as claimed by 'clang-linker-wrapper --help'.

// RUN: clang-linker-wrapper -mllvm --help 2>&1 | FileCheck %s
// RUN: clang-linker-wrapper --offload-opt=--help 2>&1 | FileCheck %s

// Look for a few options supported only after -mllvm and --offload-opt.
//     CHECK: OPTIONS:
// CHECK-DAG: --passes=<string>
// CHECK-DAG: --load-pass-plugin=<string>
