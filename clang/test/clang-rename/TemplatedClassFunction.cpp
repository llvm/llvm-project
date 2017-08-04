template <typename T>
class A {
public:
  void foo() /* Test 1 */ {}  // CHECK: void bar() /* Test 1 */ {}
};

int main(int argc, char **argv) {
  A<int> a;
  a.foo();   /* Test 2 */     // CHECK: a.bar()   /* Test 2 */
  return 0;
}

// Test 1.
// RUN: clang-refactor rename -offset=48 -new-name=bar %s -- | sed 's,//.*,,' | FileCheck %s
// Test 2.
// RUN: clang-refactor rename -offset=162 -new-name=bar %s -- | sed 's,//.*,,' | FileCheck %s
//
// Currently unsupported test.
// XFAIL: *

// To find offsets after modifying the file, use:
//   grep -Ubo 'foo.*' <file>
