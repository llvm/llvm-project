template <typename T>             // CHECK: template <typename U>
class Foo {
T foo(T arg, T& ref, T* ptr) {    // CHECK: U foo(U arg, U& ref, U* ptr) {
  T value;                        // CHECK: U value;
  int number = 42;
  value = (T)number;              // CHECK: value = (U)number;
  value = static_cast<T>(number); // CHECK: value = static_cast<U>(number);
  return value;
}

static void foo(T value) {}       // CHECK: static void foo(U value) {}

T member;                         // CHECK: U member;
};

// RUN: cat %s > %t.cpp
// RUN: clang-rename -offset=19 -new-name=U %t.cpp -i -- -fno-delayed-template-parsing
// RUN: sed 's,//.*,,' %t.cpp | FileCheck %s
