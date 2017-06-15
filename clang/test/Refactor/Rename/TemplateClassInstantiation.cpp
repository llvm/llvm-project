template <typename T>
class Foo { /* Test 1 */   // CHECK: rename [[@LINE]]:7 -> [[@LINE]]:10
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
  Foo<T> obj; /* Test 2 */  // CHECK: rename [[@LINE]]:3 -> [[@LINE]]:6
  obj.member = T();
  Foo<T>::foo();            // CHECK: rename [[@LINE]]:3 -> [[@LINE]]:6
}

int main() {
  Foo<int> i; /* Test 3 */  // CHECK: rename [[@LINE]]:3 -> [[@LINE]]:6
  i.member = 0;
  Foo<int>::foo(0);         // CHECK: rename [[@LINE]]:3 -> [[@LINE]]:6

  Foo<bool> b;              // CHECK: rename [[@LINE]]:3 -> [[@LINE]]:6
  b.member = false;
  Foo<bool>::foo(false);    // CHECK: rename [[@LINE]]:3 -> [[@LINE]]:6

  return 0;
}

// Test 1.
// RUN: clang-refactor-test rename-initiate -at=%s:2:7 -new-name=Bar %s -fno-delayed-template-parsing | FileCheck %s
// Test 2.
// RUN: clang-refactor-test rename-initiate -at=%s:17:3 -new-name=Bar %s -fno-delayed-template-parsing | FileCheck %s
// Test 3.
// RUN: clang-refactor-test rename-initiate -at=%s:25:3 -new-name=Bar %s -fno-delayed-template-parsing | FileCheck %s
