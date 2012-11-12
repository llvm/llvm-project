// REQUIRES: x86-registered-target
// RUN: %clang -target i386-apple-darwin10 -flto -S -g %s -o - | FileCheck %s
int global;
int main() { 
  static int localstatic;
  return 0;
}

// CHECK: metadata !{i32 {{.*}}, i32 0, metadata !5, metadata !"localstatic", metadata !"localstatic", metadata !"", metadata !6, i32 5, metadata !9, i32 1, i32 1, i32* @main.localstatic} ; [ DW_TAG_variable ]
// CHECK: metadata !{i32 {{.*}}, i32 0, null, metadata !"global", metadata !"global", metadata !"", metadata !6, i32 3, metadata !9, i32 0, i32 1, i32* @global} ; [ DW_TAG_variable ]
