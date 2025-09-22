// RUN: clang-reorder-fields -record-name Foo -fields-order z,y,x %s -- | FileCheck %s

struct Foo {
  int x;  // CHECK:       {{^  int z;}}
  int y;  // CHECK-NEXT:  {{^  int y;}}
  int z;  // CHECK-NEXT:  {{^  int x;}}
};

int main() {
  struct Foo foo = { 0, 1 }; // CHECK: {{^  struct Foo foo = { .y = 1, .x = 0 };}}
  return 0;
}
