// RUN: not clang-check "%s" -- -c 2>&1 | FileCheck %s

// CHECK: a type specifier is required
invalid;
