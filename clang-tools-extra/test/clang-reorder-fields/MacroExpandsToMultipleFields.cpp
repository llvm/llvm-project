// RUN: clang-reorder-fields -record-name ::bar::Foo -fields-order z,y,x %s -- | FileCheck %s

namespace bar {

#define FIELDS_DECL int x; int y; // CHECK: {{^#define FIELDS_DECL int x; int y;}}

// The order of fields should not change.
struct Foo {
  FIELDS_DECL  // CHECK:      {{^ FIELDS_DECL}}
  int z;       // CHECK-NEXT: {{^ int z;}}
};

} // end namespace bar
