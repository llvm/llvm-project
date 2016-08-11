template <typename T>
class Foo { /* Test 1 */   // CHECK: class Bar { /* Test 1 */
public:
  T foo(T arg, T& ref, T* ptr) {
    T value;
    int number = 42;
    value = (T)number;
    value = static_cast<T>(number);
    return value;
  }
  static void foo(T value) {}
  T member;
};

template <typename T>
void func() {
  Foo<T> obj; /* Test 2 */  // CHECK: Bar<T> obj;
  obj.member = T();
  Foo<T>::foo();            // CHECK: Bar<T>::foo();
}

int main() {
  Foo<int> i; /* Test 3 */  // CHECK: Bar<int> i;
  i.member = 0;
  Foo<int>::foo(0);         // CHECK: Bar<int>::foo(0);

  Foo<bool> b;              // CHECK: Bar<bool> b;
  b.member = false;
  Foo<bool>::foo(false);    // CHECK: Bar<bool>::foo(false);

  return 0;
}

// Test 1.
// RUN: clang-rename -offset=29 -new-name=Bar %s -- -fno-delayed-template-parsing | sed 's,//.*,,' | FileCheck %s
// Test 2.
// RUN: clang-rename -offset=324 -new-name=Bar %s -- -fno-delayed-template-parsing | sed 's,//.*,,' | FileCheck %s
// Test 3.
// RUN: clang-rename -offset=463 -new-name=Bar %s -- -fno-delayed-template-parsing | sed 's,//.*,,' | FileCheck %s

// To find offsets after modifying the file, use:
//   grep -Ubo 'Foo.*' <file>
