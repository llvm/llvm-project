// RUN: clang-reorder-fields -record-name Foo -fields-order z,w,y,x %s -- | FileCheck %s

struct Foo {
  const int* x; // CHECK:      {{^  double z;}}
  int y;        // CHECK-NEXT: {{^  int w;}}
  double z;     // CHECK-NEXT: {{^  int y;}}
  int w;        // CHECK-NEXT: {{^  const int\* x}}
};

int main(void) {
  const int x = 13;
  struct Foo foo = { &x, 0, 1.29, 17 }; // CHECK: {{^  struct Foo foo = { 1.29, 17, 0, &x };}} 
  return 0;
}
