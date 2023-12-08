class Baz {};

class Qux {
  Baz Foo;         /* Test 1 */       // CHECK: Baz Bar;
public:
  Qux();
};

Qux::Qux() : Foo() /* Test 2 */ {}    // CHECK: Qux::Qux() : Bar() /* Test 2 */ {}

// Test 1.
// RUN: clang-rename -offset=33 -new-name=Bar %s -- | sed 's,//.*,,' | FileCheck %s
// Test 2.
// RUN: clang-rename -offset=118 -new-name=Bar %s -- | sed 's,//.*,,' | FileCheck %s

// To find offsets after modifying the file, use:
//   grep -Ubo 'Foo.*' <file>
