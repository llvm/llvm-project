// RUN: %clang -flimit-debug-info -emit-llvm -g -S %s -o - | FileCheck %s
// RUN: %clang -flimit-debug-info -emit-llvm -g -S %s -o - | FileCheck --check-prefix=CHECK-C %s

// CHECK: !DICompositeType(tag: DW_TAG_class_type, name: "A"
// CHECK-NOT:              DIFlagFwdDecl
// CHECK-SAME:             ){{$}}
class A {
public:
  int z;
};

A *foo (A* x) {
  A *a = new A(*x);
  return a;
}

// CHECK: !DICompositeType(tag: DW_TAG_class_type, name: "B"
// CHECK-SAME:             flags: DIFlagFwdDecl

class B {
public:
  int y;
};

extern int bar(B *b);
int baz(B *b) {
  return bar(b);
}


// CHECK-C: !DICompositeType(tag: DW_TAG_structure_type, name: "C"
// CHECK-C-SAME:             flags: DIFlagFwdDecl

struct C {
};

C (*x)(C);
