// RUN: %clang_cc1 -ast-print %s -o - | FileCheck %s

// CHECK: typedef _Bool bool32 __attribute__((ext_vector_type(32)));
typedef _Bool bool32 __attribute__((ext_vector_type(32)));
