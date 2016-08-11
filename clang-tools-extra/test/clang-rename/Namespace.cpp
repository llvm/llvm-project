namespace gcc /* Test 1 */ {  // CHECK: namespace clang /* Test 1 */ {
  int x;
}

void boo() {
  gcc::x = 42;                // CHECK: clang::x = 42;
}

// Test 1.
// RUN: clang-rename -offset=10 -new-name=clang %s -- | sed 's,//.*,,' | FileCheck %s

// To find offsets after modifying the file, use:
//   grep -Ubo 'Foo.*' <file>
