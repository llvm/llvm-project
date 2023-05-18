class Foo /* Test 1 */ {};    // CHECK: class Bar /* Test 1 */ {};

template <typename T>
void func() {}

template <typename T>
class Baz {};

int main() {
  func<Foo>();                // CHECK: func<Bar>();
  Baz<Foo> /* Test 2 */ obj;  // CHECK: Baz<Bar> /* Test 2 */ obj;
  return 0;
}

// Test 1.
// RUN: clang-rename -offset=7 -new-name=Bar %s -- | sed 's,//.*,,' | FileCheck %s
// Test 2.
// RUN: clang-rename -offset=215 -new-name=Bar %s -- | sed 's,//.*,,' | FileCheck %s

// To find offsets after modifying the file, use:
//   grep -Ubo 'Foo.*' <file>
