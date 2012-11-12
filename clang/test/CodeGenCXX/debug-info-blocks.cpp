// RUN: %clang_cc1 %s -gline-tables-only -fblocks -S -emit-llvm -o - | FileCheck %s

struct A {
  A();
  A(const A &);
  ~A();
};

void test() {
  __block A a;
}

// CHECK: [ DW_TAG_subprogram ] [line 10] [local] [def] [__Block_byref_object_copy_]
// CHECK: [ DW_TAG_subprogram ] [line 10] [local] [def] [__Block_byref_object_dispose_]
