#define Baz Foo // CHECK: #define Baz Bar

void foo(int value) {}

void macro() {
  int Foo;  /* Test 1 */  // CHECK: int Bar;
  Foo = 42; /* Test 2 */  // CHECK: Bar = 42;
  Baz -= 0;
  foo(Foo); /* Test 3 */  // CHECK: foo(Bar);
  foo(Baz);
}

// Test 1.
// RUN: clang-rename -offset=88 -new-name=Bar %s -- | sed 's,//.*,,' | FileCheck %s
// Test 2.
// RUN: clang-rename -offset=129 -new-name=Bar %s -- | sed 's,//.*,,' | FileCheck %s
// Test 3.
// RUN: clang-rename -offset=191 -new-name=Bar %s -- | sed 's,//.*,,' | FileCheck %s

// To find offsets after modifying the file, use:
//   grep -Ubo 'Foo.*' <file>
