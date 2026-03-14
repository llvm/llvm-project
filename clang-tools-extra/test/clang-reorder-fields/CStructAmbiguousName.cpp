// RUN: clang-reorder-fields -record-name ::Foo -fields-order y,x %s -- | FileCheck %s

struct Foo {
  int x;    // CHECK:      {{^  double y;}}
  double y; // CHECK-NEXT: {{^  int x;}}
};

namespace bar {
struct Foo {
  int x;    // CHECK:      {{^  int x;}}
  double y; // CHECK-NEXT: {{^  double y;}}
};
} // end namespace bar

int main() {
  bar::Foo foo = { 1, 1.7 }; // CHECK: {{^  bar::Foo foo = { 1, 1.7 };}}
  return 0;
}
