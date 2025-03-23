// RUN: clang-reorder-fields -record-name Foo -fields-order y,x %s -- | FileCheck %s

#define DEFINE_FIELD(name) \
    int name

// We should not reorder given that the definition of `x` is in a macro
// expansion.
class Foo {
  DEFINE_FIELD(x); // CHECK:      DEFINE_FIELD(x);
  int y;           // CHECK-NEXT: int y;
};

