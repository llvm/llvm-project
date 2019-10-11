// RUN: not clang-check "%s" -- -c -x hip -nogpulib 2>&1 | FileCheck %s

// CHECK: C++ requires
invalid;
