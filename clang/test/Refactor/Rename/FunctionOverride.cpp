class A { virtual void foo();     /* Test 1 */ }; // CHECK: rename [[@LINE]]:24 -> [[@LINE]]:27
class B : public A { void foo();  /* Test 2 */ }; // CHECK: rename [[@LINE]]:27 -> [[@LINE]]:30
class C : public B { void foo();  /* Test 3 */ }; // CHECK: rename [[@LINE]]:27 -> [[@LINE]]:30

// Test 1.
// RUN: clang-refactor-test rename-initiate -at=%s:1:24 -new-name=bar %s | FileCheck %s
// Test 2.
// RUN: clang-refactor-test rename-initiate -at=%s:2:27 -new-name=bar %s | FileCheck %s
// Test 3.
// RUN: clang-refactor-test rename-initiate -at=%s:3:27 -new-name=bar %s | FileCheck %s
