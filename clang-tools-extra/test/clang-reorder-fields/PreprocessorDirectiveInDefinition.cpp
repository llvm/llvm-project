// RUN: clang-reorder-fields -record-name ::bar::Foo -fields-order z,y,x %s -- | FileCheck %s

namespace bar {

#define ADD_Z

// The order of fields should not change.
struct Foo {
  int x;     // CHECK:      {{^ int x;}}
  int y;     // CHECK-NEXT: {{^ int y;}}
#ifdef ADD_Z // CHECK-NEXT: {{^#ifdef ADD_Z}}
  int z;     // CHECK-NEXT: {{^ int z;}}
#endif       // CHECK-NEXT: {{^#endif}}
};

} // end namespace bar
