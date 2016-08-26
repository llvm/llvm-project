class Baz {
  int Foo; /* Test 1 */ // CHECK: int Bar;
public:
  Baz();
};

Baz::Baz() : Foo(0) /* Test 2 */ {}  // CHECK: Baz::Baz() : Bar(0)

// Test 1.
// RUN: clang-rename -offset=18 -new-name=Bar %s -- | sed 's,//.*,,' | FileCheck %s
// Test 2.
// RUN: clang-rename -offset=89 -new-name=Bar %s -- | sed 's,//.*,,' | FileCheck %s

// To find offsets after modifying the file, use:
//   grep -Ubo 'Foo.*' <file>
