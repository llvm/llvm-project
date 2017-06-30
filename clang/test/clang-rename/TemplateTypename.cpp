template <typename T /* Test 1 */>              // CHECK: template <typename U /* Test 1 */>
class Foo {
T foo(T arg, T& ref, T* /* Test 2 */ ptr) {     // CHECK: U foo(U arg, U& ref, U* /* Test 2 */ ptr) {
  T value;                                      // CHECK: U value;
  int number = 42;
  value = (T)number;                            // CHECK: value = (U)number;
  value = static_cast<T /* Test 3 */>(number);  // CHECK: value = static_cast<U /* Test 3 */>(number);
  return value;
}

static void foo(T value) {}                     // CHECK: static void foo(U value) {}

T member;                                       // CHECK: U member;
};

// Test 1.
// RUN: clang-rename -offset=19 -new-name=U %s -- -fno-delayed-template-parsing | sed 's,//.*,,' | FileCheck %s
// Test 2.
// RUN: clang-rename -offset=126 -new-name=U %s -- -fno-delayed-template-parsing | sed 's,//.*,,' | FileCheck %s
// Test 3.
// RUN: clang-rename -offset=392 -new-name=U %s -- -fno-delayed-template-parsing | sed 's,//.*,,' | FileCheck %s

// To find offsets after modifying the file, use:
//   grep -Ubo 'T.*' <file>
