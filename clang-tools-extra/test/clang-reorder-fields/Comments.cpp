// RUN: clang-reorder-fields -record-name Foo -fields-order e1,e3,e2,a,c,b %s -- | FileCheck %s

class Foo {
  int a; // Trailing comment for a.
  int b; // Multiline
         // trailing for b.
  // Prefix comments for c.
  int c;

  /*c-like*/ int e1;
  int /*c-like*/ e2;
  int e3 /*c-like*/;
};

// CHECK:       /*c-like*/ int e1;
// CHECK-NEXT:  int e3 /*c-like*/;
// CHECK-NEXT:  int /*c-like*/ e2;
// CHECK-NEXT:  int a; // Trailing comment for a.
// CHECK-NEXT:  // Prefix comments for c.
// CHECK-NEXT:  int c;
// CHECK-NEXT:  int b; // Multiline
// CHECK-NEXT:         // trailing for b.

