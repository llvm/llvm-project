class Foo {       /* Test 1 */          // CHECK: class Bar {
public:
  Foo() {}                              // CHECK: Bar() {}
};

class Baz {
public:
  operator Foo()  /* Test 2 */ const {  // CHECK: operator Bar()  /* Test 2 */ const {
    Foo foo;                            // CHECK: Bar foo;
    return foo;
  }
};

int main() {
  Baz boo;
  Foo foo = static_cast<Foo>(boo);      // CHECK: Bar foo = static_cast<Bar>(boo);
  return 0;
}

// Test 1.
// RUN: clang-rename -offset=7 -new-name=Bar %s -- | sed 's,//.*,,' | FileCheck %s
// Test 2.
// RUN: clang-rename -offset=164 -new-name=Bar %s -- | sed 's,//.*,,' | FileCheck %s

// To find offsets after modifying the file, use:
//   grep -Ubo 'Foo.*' <file>
