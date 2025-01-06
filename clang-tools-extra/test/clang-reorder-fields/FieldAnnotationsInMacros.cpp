// RUN: clang-reorder-fields -record-name Foo -fields-order y,x %s -- | FileCheck %s

#define GUARDED_BY(x) __attribute__((guarded_by(x)))

class Foo {
  int x GUARDED_BY(x); // CHECK: {{^  int y;}}
  int y;               // CHECK-NEXT: {{^  int x GUARDED_BY\(x\);}}
};

