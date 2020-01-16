// UNSUPPORTED: system-windows

// RUN: not --crash clang-tidy -p %S/Inputs/empty-database %s 2>&1 | FileCheck %s

// CHECK: LLVM ERROR: Cannot chdir into ""!
