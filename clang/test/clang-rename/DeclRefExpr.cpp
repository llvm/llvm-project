class C {
public:
  static int Foo; /* Test 1 */  // CHECK: static int Bar;
};

int foo(int x) { return 0; }
#define MACRO(a) foo(a)

int main() {
  C::Foo = 1;     /* Test 2 */  // CHECK: C::Bar = 1;
  MACRO(C::Foo);                // CHECK: MACRO(C::Bar);
  int y = C::Foo; /* Test 3 */  // CHECK: int y = C::Bar;
  return 0;
}

// Test 1.
// RUN: clang-rename -offset=31 -new-name=Bar %s -- | sed 's,//.*,,' | FileCheck %s
// Test 2.
// RUN: clang-rename -offset=152 -new-name=Bar %s -- | sed 's,//.*,,' | FileCheck %s
// Test 3.
// RUN: clang-rename -offset=271 -new-name=Bar %s -- | sed 's,//.*,,' | FileCheck %s

// To find offsets after modifying the file, use:
//   grep -Ubo 'Foo.*' <file>
