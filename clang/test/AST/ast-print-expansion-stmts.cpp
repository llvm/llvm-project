// Without serialization:
// RUN: %clang_cc1 -std=c++26 -ast-print %s | FileCheck %s
//
// With serialization:
// RUN: %clang_cc1 -std=c++26 -emit-pch -o %t %s
// RUN: %clang_cc1 -x c++ -std=c++26 -include-pch %t -ast-print  /dev/null | FileCheck %s

template <typename T, __SIZE_TYPE__ size>
struct Array {
  T data[size]{};
  constexpr const T* begin() const { return data; }
  constexpr const T* end() const { return data + size; }
};

// CHECK: void foo(int);
void foo(int);

// CHECK: template <typename T> void test(T t) {
template <typename T>
void test(T t) {
  // Enumerating expansion statement.
  //
  // CHECK:      template for (auto x : { 1, 2, 3 }) {
  // CHECK-NEXT:     foo(x);
  // CHECK-NEXT: }
  template for (auto x : {1, 2, 3}) {
    foo(x);
  }

  // Iterating expansion statement.
  //
  // CHECK:      static constexpr Array<int, 3> a;
  // CHECK-NEXT: template for (auto x : a) {
  // CHECK-NEXT:   foo(x);
  // CHECK-NEXT: }
  static constexpr Array<int, 3> a;
  template for (auto x : a) {
    foo(x);
  }

  // Destructuring expansion statement.
  //
  // CHECK:      int arr[3]{1, 2, 3};
  // CHECK-NEXT: template for (auto x : arr) {
  // CHECK-NEXT:   foo(x);
  // CHECK-NEXT: }
  int arr[3]{1, 2, 3};
  template for (auto x : arr) {
    foo(x);
  }

  // Dependent expansion statement.
  //
  // CHECK:      template for (auto x : t) {
  // CHECK-NEXT:   foo(x);
  // CHECK-NEXT: }
  template for (auto x : t) {
    foo(x);
  }
}

// CHECK: template <typename T> void test2(T t) {
template <typename T>
void test2(T t) {
  // Enumerating expansion statement.
  //
  // CHECK:      template for (int x : { 1, 2, 3 }) {
  // CHECK-NEXT:     foo(x);
  // CHECK-NEXT: }
  template for (int x : {1, 2, 3}) {
    foo(x);
  }

  // Iterating expansion statement.
  //
  // CHECK:      static constexpr Array<int, 3> a;
  // CHECK-NEXT: template for (int x : a) {
  // CHECK-NEXT:   foo(x);
  // CHECK-NEXT: }
  static constexpr Array<int, 3> a;
  template for (int x : a) {
    foo(x);
  }

  // Destructuring expansion statement.
  //
  // CHECK:      int arr[3]{1, 2, 3};
  // CHECK-NEXT: template for (int x : arr) {
  // CHECK-NEXT:   foo(x);
  // CHECK-NEXT: }
  int arr[3]{1, 2, 3};
  template for (int x : arr) {
    foo(x);
  }

  // Dependent expansion statement.
  //
  // CHECK:      template for (int x : t) {
  // CHECK-NEXT:   foo(x);
  // CHECK-NEXT: }
  template for (int x : t) {
    foo(x);
  }
}
