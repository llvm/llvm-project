template <typename T /* Test 1 */>              // CHECK: rename local [[@LINE]]:20 -> [[@LINE]]:21
class Foo {
T foo(T arg, T& ref, T* /* Test 2 */ ptr) {     // CHECK: rename local [[@LINE]]:1 -> [[@LINE]]:2
                                                // CHECK: rename local [[@LINE-1]]:7 -> [[@LINE-1]]:8
                                                // CHECK: rename local [[@LINE-2]]:14 -> [[@LINE-2]]:15
                                                // CHECK: rename local [[@LINE-3]]:22 -> [[@LINE-3]]:23
  T value;                                      // CHECK: rename local [[@LINE]]:3 -> [[@LINE]]:4
  int number = 42;
  value = (T)number;                            // CHECK: rename local [[@LINE]]:12 -> [[@LINE]]:13
  value = static_cast<T /* Test 3 */>(number);  // CHECK: rename local [[@LINE]]:23 -> [[@LINE]]:24
  return value;
}

static void foo(T value) {}                     // CHECK: rename local [[@LINE]]:17 -> [[@LINE]]:18

T member;                                       // CHECK: rename local [[@LINE]]:1 -> [[@LINE]]:2
};

// Test 1.
// RUN: clang-refactor-test rename-initiate -at=%s:1:20 -new-name=U %s -fno-delayed-template-parsing | FileCheck %s
// Test 2.
// RUN: clang-refactor-test rename-initiate -at=%s:3:22 -new-name=U %s -fno-delayed-template-parsing | FileCheck %s
// Test 3.
// RUN: clang-refactor-test rename-initiate -at=%s:10:23 -new-name=U %s -fno-delayed-template-parsing | FileCheck %s
