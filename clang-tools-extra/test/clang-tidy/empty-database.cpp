// RUN: not clang-tidy -p %S/empty-database %s 2>&1 | FileCheck %s

// CHECK: LLVM ERROR: Cannot chdir into ""!
