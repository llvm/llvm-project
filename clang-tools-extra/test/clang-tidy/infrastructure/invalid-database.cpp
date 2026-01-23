// UNSUPPORTED: system-windows

// RUN: not --crash clang-tidy -checks='-*,clang-analyzer-*' -p %S/Inputs/invalid-database %s 2>&1 | FileCheck %s

// CHECK: LLVM ERROR: Cannot chdir into "/invalid/"!
