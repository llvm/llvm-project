// RUN: %clang_cc1 -emit-llvm -std=c++11 -g %s -o -| FileCheck %s

// 16 is DW_ATE_UTF (0x10) encoding attribute.
char16_t char_a = u'h';

// CHECK: !7 = metadata !{i32 {{.*}}, null, metadata !"char16_t", null, i32 0, i64 16, i64 16, i64 0, i32 0, i32 16} ; [ DW_TAG_base_type ]
