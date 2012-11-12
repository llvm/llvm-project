// RUN: %clang_cc1 -emit-llvm -g -triple x86_64-apple-darwin %s -o - | FileCheck %s

// CHECK: metadata !{i32 {{.*}}, i64 1, i64 0}        ; [ DW_TAG_subrange_type ]

struct StructName {
  int member[];
};

struct StructName SN;
