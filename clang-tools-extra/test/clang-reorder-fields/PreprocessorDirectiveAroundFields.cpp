// RUN: clang-reorder-fields -record-name ::bar::Foo -fields-order y,x %s -- | FileCheck %s

namespace bar {

#define DEFINE_FIELDS

// This is okay to reorder.
struct Foo {
#ifdef DEFINE_FIELDS // CHECK:      {{^#ifdef DEFINE_FIELDS}}
  int x;             // CHECK-NEXT: {{^ int y;}}
  int y;             // CHECK-NEXT: {{^ int x;}}
#endif               // CHECK-NEXT: {{^#endif}}
};

} // end namespace bar
