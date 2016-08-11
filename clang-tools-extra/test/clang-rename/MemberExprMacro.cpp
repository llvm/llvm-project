class Baz {
public:
  int Foo;  /* Test 1 */    // CHECK: int Bar;
};

int qux(int x) { return 0; }
#define MACRO(a) qux(a)

int main() {
  Baz baz;
  baz.Foo = 1; /* Test 2 */ // CHECK: baz.Bar = 1;
  MACRO(baz.Foo);           // CHECK: MACRO(baz.Bar);
  int y = baz.Foo;          // CHECK: int y = baz.Bar;
}

// Test 1.
// RUN: clang-rename -offset=26 -new-name=Bar %s -- | sed 's,//.*,,' | FileCheck %s
// Test 2.
// RUN: clang-rename -offset=155 -new-name=Bar %s -- | sed 's,//.*,,' | FileCheck %s

// To find offsets after modifying the file, use:
//   grep -Ubo 'Foo.*' <file>
