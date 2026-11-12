// RUN: clang-reorder-fields -record-name ::bar::Foo -fields-order z,w,y,x %s -- | FileCheck %s

namespace bar {
struct Foo {
  const int* x; // CHECK:      {{^  double z;}}
  int y;        // CHECK-NEXT: {{^  int w;}}
  double z;     // CHECK-NEXT: {{^  int y;}}
  int w;        // CHECK-NEXT: {{^  const int\* x}}
};
} // end namespace bar

int main() {
  const int x = 13;
  bar::Foo foo = { &x, 0, 1.29, 17 }; // CHECK: {{^  bar::Foo foo = { 1.29, 17, 0, &x };}} 
  return 0;
}
