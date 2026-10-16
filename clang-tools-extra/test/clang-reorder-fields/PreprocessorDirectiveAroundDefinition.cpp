// RUN: clang-reorder-fields -record-name ::bar::Foo -fields-order y,x %s -- | FileCheck %s

namespace bar {

#define DEFINE_FOO

// This is okay to reorder.
#ifdef DEFINE_FOO
struct Foo {
  int x;     // CHECK:      {{^ int y;}}
  int y;     // CHECK-NEXT: {{^ int x;}}
};
#endif

} // end namespace bar
