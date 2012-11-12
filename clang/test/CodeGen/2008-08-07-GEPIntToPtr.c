// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s
// PR2603

struct A {
  char num_fields;
};

struct B {
  char a, b[1];
};

const struct A Foo = {
  // CHECK: i8 1
  (char *)(&( (struct B *)(16) )->b[0]) - (char *)(16)
};
