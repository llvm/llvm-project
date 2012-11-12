// RUN: %clang_cc1 -std=c++11 -emit-llvm -g -triple x86_64-apple-darwin %s -o - | FileCheck %s

extern "C" {
extern int printf(const char * format, ...);
}
void foo (int &&i)
{
  printf("%d\n", i);
}

// CHECK: metadata !{i32 {{.*}}, null, null, null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !10} ; [ DW_TAG_rvalue_reference_type ]
