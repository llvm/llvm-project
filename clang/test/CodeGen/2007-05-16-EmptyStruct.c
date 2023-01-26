// PR 1417
// RUN: %clang_cc1   %s -emit-llvm -o - | FileCheck %s

// CHECK: global ptr null
struct { } *X;
