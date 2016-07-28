// RUN: clang-rename -offset=289 -new-name=U %s -- | FileCheck %s

// Currently unsupported test.
// FIXME: clang-rename should be able to rename template parameters correctly.
// XFAIL: *

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
