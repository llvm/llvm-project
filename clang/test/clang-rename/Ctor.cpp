class Foo {                   // CHECK: class Bar {
public:
  Foo();    /* Test 1 */      // CHECK: Bar();
};

Foo::Foo()  /* Test 2 */ {}   // CHECK: Bar::Bar()  /* Test 2 */ {}

// Test 1.
// RUN: clang-rename -offset=62 -new-name=Bar %s -- | sed 's,//.*,,' | FileCheck %s
// Test 2.
// RUN: clang-rename -offset=116 -new-name=Bar %s -- | sed 's,//.*,,' | FileCheck %s

// To find offsets after modifying the file, use:
//   grep -Ubo 'Foo.*' <file>
