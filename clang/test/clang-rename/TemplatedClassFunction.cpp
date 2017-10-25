template <typename T>
class A {
public:
  void foo() /* Test 1 */ {}  // CHECK: void bar() /* Test 1 */ {}
};

int main(int argc, char **argv) {
  A<int> a;
  A<double> b;
  A<float> c;
  a.foo();   /* Test 2 */     // CHECK: a.bar();   /* Test 2 */
  b.foo();   /* Test 3 */     // CHECK: b.bar();   /* Test 3 */
  c.foo();   /* Test 4 */     // CHECK: c.bar();   /* Test 4 */
  return 0;
}

// Test 1.
// RUN: clang-rename -offset=48 -new-name=bar %s -- | sed 's,//.*,,' | FileCheck %s
// Test 2.
// RUN: clang-rename -offset=191 -new-name=bar %s -- | sed 's,//.*,,' | FileCheck %s
// Test 3.
// RUN: clang-rename -offset=255 -new-name=bar %s -- | sed 's,//.*,,' | FileCheck %s
// Test 4.
// RUN: clang-rename -offset=319 -new-name=bar %s -- | sed 's,//.*,,' | FileCheck %s

// To find offsets after modifying the file, use:
//   grep -Ubo 'foo.*' <file>
