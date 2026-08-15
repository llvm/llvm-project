// RUN: %check_clang_tidy -std=c++17-or-later %s bugprone-smart-ptr-initialization %t -- -- -I %S/../Inputs/Headers/std

#include <memory>
#include <utility>

struct A {
  int x;
};

A arr[10];

void test_shared_ptr_constructor() {
  std::shared_ptr<A[]> a(arr);
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: passing a raw pointer 'A[10]' to 'std::shared_ptr<A[]>' constructor may cause double deletion
}

void test_stack_variable() {
  int x[10] = {5};
  std::shared_ptr<int[]> ptr(x);
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: passing a raw pointer 'int[10]' to 'std::shared_ptr<int[]>' constructor may cause double deletion
}

// Should trigger for member variables
struct S {
  int member[10];
  void test() {
    std::shared_ptr<int[]> ptr(member);
    // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: passing a raw pointer 'int[10]' to 'std::shared_ptr<int[]>' constructor may cause double deletion
  }
};

void test_new_expression_ok() {
  std::shared_ptr<A[]> a(new A[10]);
}

struct NoopDeleter {
    void operator() (A* p) {}
};

void test_custom_deleter_ok() {
  auto noop_deleter = [](A* p) {  };
  std::shared_ptr<A[]> p2(arr, noop_deleter);
}

void test_nullptr_ok() {
  std::shared_ptr<A[]> a(nullptr);
}

void test_zero_ok() {
  std::shared_ptr<A[]> a(0);
}

void test_copy_move_constructor_ok(std::shared_ptr<A[]> sp) {
  auto sp2 = sp;
  auto sp3 = std::move(sp);
}

void test_shared_ptr_reset() {
  std::shared_ptr<A[]> a;
  a.reset(arr);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: passing a raw pointer 'A[10]' to 'std::shared_ptr<A[]>::reset' may cause double deletion
}

void test_stack_variable_reset() {
  int x[10] = {5};
  std::shared_ptr<int[]> ptr;
  ptr.reset(x);
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: passing a raw pointer 'int[10]' to 'std::shared_ptr<int[]>::reset' may cause double deletion
}

void test_new_expression_reset_ok() {
  std::shared_ptr<A[]> a;
  a.reset(new A[10]);
}

void test_custom_deleter_reset_ok() {
  auto noop_deleter = [](A* p) {  };
  std::shared_ptr<A[]> p2;
  // FIXME: mock shared_ptr must support reset with custom deleter
  // p2.reset(arr, noop_deleter);
}

void test_reset_ok() {
  std::shared_ptr<A[]> a;
  a.reset();
}

// Edge case: should trigger for array new with wrong smart pointer
void test_array_new() {
  std::shared_ptr<A[]> sp(new A); // This is actually wrong but not our check's concern
  sp.reset(new A);
  // This would be caught by bugprone-shared-ptr-array-mismatch checks
}
