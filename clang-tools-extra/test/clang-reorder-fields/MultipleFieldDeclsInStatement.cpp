// RUN: clang-reorder-fields -record-name ::bar::Foo -fields-order z,y,x %s -- | FileCheck %s

namespace bar {

// The order of fields should not change.
struct Foo {
  int x, y; // CHECK: {{^  int x, y;}}
  double z; // CHECK-NEXT: {{^  double z;}}
};

} // end namespace bar
