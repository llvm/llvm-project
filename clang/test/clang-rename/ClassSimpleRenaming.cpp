class Foo /* Test 1 */ {              // CHECK: class Bar /* Test 1 */ {
public:
  void foo(int x);
};

void Foo::foo(int x) /* Test 2 */ {}  // CHECK: void Bar::foo(int x) /* Test 2 */ {}

// Test 1.
// RUN: clang-rename -offset=6 -new-name=Bar %s -- | sed 's,//.*,,' | FileCheck %s
// Test 2.
// RUN: clang-rename -offset=109 -new-name=Bar %s -- | sed 's,//.*,,' | FileCheck %s

// To find offsets after modifying the file, use:
//   grep -Ubo 'Foo.*' <file>
