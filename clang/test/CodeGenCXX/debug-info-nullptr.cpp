// RUN: %clang_cc1 -emit-llvm -std=c++11 -g %s -o -| FileCheck %s

void foo() {
  decltype(nullptr) t = 0;
}

// CHECK: metadata !{i32 {{.*}}, null, metadata !"nullptr_t", null, i32 0, i64 0, i64 0, i64 0, i32 0, i32 0} ; [ DW_TAG_unspecified_type ]
