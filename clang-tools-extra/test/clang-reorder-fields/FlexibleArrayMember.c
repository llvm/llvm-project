// RUN: clang-reorder-fields -record-name Foo -fields-order z,y,x %s -- 2>&1 | FileCheck --check-prefix=CHECK-BAD %s
// RUN: clang-reorder-fields -record-name Foo -fields-order y,x,z %s -- | FileCheck --check-prefix=CHECK-GOOD %s

// CHECK-BAD: {{^Flexible array member must remain the last field in the struct}}

struct Foo {
  int x;   // CHECK-GOOD:      {{^  int y;}}
  int y;   // CHECK-GOOD-NEXT: {{^  int x;}}
  int z[]; // CHECK-GOOD-NEXT: {{^  int z\[\];}}
};
