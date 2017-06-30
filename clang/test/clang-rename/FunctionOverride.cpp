class A { virtual void foo();     /* Test 1 */ }; // CHECK: class A { virtual void bar();
class B : public A { void foo();  /* Test 2 */ }; // CHECK: class B : public A { void bar();
class C : public B { void foo();  /* Test 3 */ }; // CHECK: class C : public B { void bar();

// Test 1.
// RUN: clang-rename -offset=23 -new-name=bar %s -- | sed 's,//.*,,' | FileCheck %s
// Test 2.
// RUN: clang-rename -offset=116 -new-name=bar %s -- | sed 's,//.*,,' | FileCheck %s
// Test 3.
// RUN: clang-rename -offset=209 -new-name=bar %s -- | sed 's,//.*,,' | FileCheck %s

// To find offsets after modifying the file, use:
//   grep -Ubo 'foo.*' <file>
