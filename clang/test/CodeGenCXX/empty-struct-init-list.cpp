// RUN: %clang_cc1 -std=c++11 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -std=c++14 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -std=c++17 -emit-llvm -o - %s | FileCheck %s

// CHECK: struct.a
typedef struct { } a;
typedef struct {
  a b[];
} c;

// CHECK: {{(dso_local )?}}global %struct.c{{.*}}zeroinitializer
c d{ };
