// RUN: %clang_cc1 %s -triple hexagon-unknown-elf -emit-llvm -o - 2>&1 | FileCheck %s
// CHECK-NOT: '+' is not a recognized feature for this target

// Empty
