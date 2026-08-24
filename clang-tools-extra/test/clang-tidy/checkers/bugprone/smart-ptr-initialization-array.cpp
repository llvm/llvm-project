// RUN: %check_clang_tidy -std=c++11-or-later %s bugprone-smart-ptr-initialization %t -- -- -I %S/../Inputs/Headers/std

#include <memory>
#include <utility>

struct A {
  int x;
};

A arr[10];

void test_unique_ptr_constructor() {
  std::unique_ptr<A[]> b(arr);
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: passing a raw pointer 'A[10]' to 'std::unique_ptr<A[]>' constructor may cause double deletion
}

void test_stack_variable() {
  int x[10] = {5};
  std::unique_ptr<int[]> ptr(x);
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: passing a raw pointer 'int[10]' to 'std::unique_ptr<int[]>' constructor may cause double deletion
}

// Should trigger for member variables
struct S {
  int member[10];
  void test() {
    std::unique_ptr<int[]> ptr(member);
    // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: passing a raw pointer 'int[10]' to 'std::unique_ptr<int[]>' constructor may cause double deletion
  }
};

void test_new_expression_ok() {
  std::unique_ptr<A[]> b(new A[10]);
}

// FIXME: WTF with our mock unique_ptr?
// void test_release_ok(std::unique_ptr<A[]> p1) {
//   std::unique_ptr<A[]> p2(p1.release());
// }

struct NoopDeleter {
    void operator() (A* p) {}
};

void test_custom_deleter_ok() {
  auto noop_deleter = [](A* p) {  };
  std::unique_ptr<A[], NoopDeleter> p0(arr);
  std::unique_ptr<A[], decltype(noop_deleter)> p1(arr, noop_deleter);
}

void test_nullptr_ok() {
  std::unique_ptr<A[]> b(nullptr);
}

void test_zero_ok() {
  std::unique_ptr<A[]> b(0);
}

void test_copy_move_constructor_ok(std::unique_ptr<A[]> up) {
  auto up3 = std::move(up);
}

void test_unique_ptr_reset() {
  std::unique_ptr<A[]> b;
  b.reset(arr);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: passing a raw pointer 'A[10]' to 'std::unique_ptr<A[]>::reset' may cause double deletion
}

void test_stack_variable_reset() {
  int x[10] = {5};
  std::unique_ptr<int[]> ptr;
  ptr.reset(x);
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: passing a raw pointer 'int[10]' to 'std::unique_ptr<int[]>::reset' may cause double deletion
}

void test_new_expression_reset_ok() {
  std::unique_ptr<A[]> b;
  b.reset(new A[10]);
}

// FIXME: WTF with our mock unique_ptr?
// void test_release_reset_ok(std::unique_ptr<A[]> p1) {
//   std::unique_ptr<A[]> p2;
//   p2.reset(p1.release());
// }

void test_custom_deleter_reset_ok() {
  auto noop_deleter = [](A* p) {  };
  std::unique_ptr<A[], NoopDeleter> p0;
  p0.reset(arr);
  std::unique_ptr<A[], decltype(noop_deleter)> p1;
  p1.reset(arr);
}

void test_nullptr_reset_ok() {
  std::unique_ptr<A[]> b;
  b.reset(nullptr);
}

void test_zero_reset_ok() {
  std::unique_ptr<A[]> b;
  b.reset(0);
}

// Edge case: should trigger for array new with wrong smart pointer
void test_array_new() {
  std::unique_ptr<A[]> sp(new A); // This is actually wrong but not our check's concern
  sp.reset(new A);
  // This would be caught by bugprone-shared-ptr-array-mismatch checks
}

void test_array_release(std::unique_ptr<A> spa) {
  std::unique_ptr<A[]> sp(spa.release()); // This is actually wrong but not our check's concern
  sp.reset(spa.release());
  // This would be caught by bugprone-shared-ptr-array-mismatch checks (mismatched new/delete)
}
