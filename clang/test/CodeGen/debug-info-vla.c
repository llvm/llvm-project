// RUN: %clang_cc1 -emit-llvm -g -triple x86_64-apple-darwin %s -o - | FileCheck %s

// CHECK: metadata !{i32 {{.*}}, metadata {{.*}}, metadata !"vla", metadata {{.*}}, i32 7, metadata {{.*}}, i32 0, i32 0, i64 2} ; [ DW_TAG_auto_variable ]

void testVLAwithSize(int s)
{
  int vla[s];
  int i;
  for (i = 0; i < s; i++) {
    vla[i] = i*i;
  }
}
