#define NAMESPACE namespace A
NAMESPACE {
int Foo;          /* Test 1 */        // CHECK: int Bar;
}
int Foo;                              // CHECK: int Foo;
int Qux = Foo;                        // CHECK: int Qux = Foo;
int Baz = A::Foo; /* Test 2 */        // CHECK: Baz = A::Bar;
void fun() {
  struct {
    int Foo;                          // CHECK: int Foo;
  } b = {100};
  int Foo = 100;                      // CHECK: int Foo = 100;
  Baz = Foo;                          // CHECK: Baz = Foo;
  {
    extern int Foo;                   // CHECK: extern int Foo;
    Baz = Foo;                        // CHECK: Baz = Foo;
    Foo = A::Foo /* Test 3 */ + Baz;  // CHECK: Foo = A::Bar /* Test 3 */ + Baz;
    A::Foo /* Test 4 */ = b.Foo;      // CHECK: A::Bar /* Test 4 */ = b.Foo;
  }
  Foo = b.Foo;                        // Foo = b.Foo;
}

// Test 1.
// RUN: clang-rename -offset=46 -new-name=Bar %s -- | sed 's,//.*,,' | FileCheck %s
// Test 2.
// RUN: clang-rename -offset=234 -new-name=Bar %s -- | sed 's,//.*,,' | FileCheck %s
// Test 3.
// RUN: clang-rename -offset=641 -new-name=Bar %s -- | sed 's,//.*,,' | FileCheck %s
// Test 4.
// RUN: clang-rename -offset=716 -new-name=Bar %s -- | sed 's,//.*,,' | FileCheck %s

// To find offsets after modifying the file, use:
//   grep -Ubo 'Foo.*' <file>
