// PR 1417
// RUN: %clang_cc1 -no-opaque-pointers   %s -emit-llvm -o - | FileCheck %s

// CHECK: global %struct.anon* null
struct { } *X;
