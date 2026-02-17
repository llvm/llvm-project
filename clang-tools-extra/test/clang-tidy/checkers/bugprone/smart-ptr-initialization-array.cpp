// RUN: %check_clang_tidy -std=c++11-or-later %s bugprone-smart-ptr-initialization %t -- -- -I%S

#include "Inputs/smart-ptr-initialization/std_smart_ptr.h"

struct A {
  int x;
};

A arr[10];

// Should trigger the check for unique_ptr constructor  
void test_unique_ptr_constructor() {
  std::unique_ptr<A[]> b(arr);
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: passing a raw pointer 'arr' to std::unique_ptr constructor may cause double deletion [bugprone-smart-ptr-initialization]
}

// Should trigger for stack variables
void test_stack_variable() {
  int x[10] = {5};
  std::unique_ptr<int[]> ptr(x);
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: passing a raw pointer 'x' to std::unique_ptr constructor may cause double deletion [bugprone-smart-ptr-initialization]
}

// Should trigger for member variables
struct S {
  int member[10];
  void test() {
    std::unique_ptr<int[]> ptr(member);
    // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: passing a raw pointer 'this->member' to std::unique_ptr constructor may cause double deletion [bugprone-smart-ptr-initialization]
  }
};

// Should NOT trigger for new expressions - these are OK
void test_new_expression_ok() {
  std::unique_ptr<A[]> b(new A[10]);
}

// Should NOT trigger for release() calls - ownership transfer
void test_release_ok(std::unique_ptr<A[]> p1) {
  std::unique_ptr<A[]> p2(p1.release());
}

struct NoopDeleter {
    void operator() (A* p) {}
};

// Should NOT trigger for custom deleters
void test_custom_deleter_ok() {
  auto noop_deleter = [](A* p) {  };
  std::unique_ptr<A[], NoopDeleter> p0(arr);
  std::unique_ptr<A[], decltype(noop_deleter)> p1(arr, noop_deleter);
}

// Should NOT trigger for nullptr
void test_nullptr_ok() {
  std::unique_ptr<A[]> b(nullptr);
}

// Should NOT trigger for copy and move constructors
void test_copy_move_constructor_ok(std::unique_ptr<A[]> up) {
  auto up3 = std::move(up);
}

// Should trigger the check for unique_ptr reset
void test_unique_ptr_reset() {
  std::unique_ptr<A[]> b;
  b.reset(arr);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: passing a raw pointer 'arr' to std::unique_ptr::reset() may cause double deletion [bugprone-smart-ptr-initialization]
}

// Should trigger for stack variables with reset
void test_stack_variable_reset() {
  int x[10] = {5};
  std::unique_ptr<int[]> ptr;
  ptr.reset(x);
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: passing a raw pointer 'x' to std::unique_ptr::reset() may cause double deletion [bugprone-smart-ptr-initialization]
}

// Should NOT trigger for new expressions with reset - these are OK
void test_new_expression_reset_ok() {
  std::unique_ptr<A[]> b;
  b.reset(new A[10]);
}

// Should NOT trigger for release() calls with reset - ownership transfer
void test_release_reset_ok(std::unique_ptr<A[]> p1) {
  std::unique_ptr<A[]> p2;
  p2.reset(p1.release());
}

// Should NOT trigger for custom deleters with reset
void test_custom_deleter_reset_ok() {
  auto noop_deleter = [](A* p) {  };
  std::unique_ptr<A[], NoopDeleter> p0;
  p0.reset(arr);
  std::unique_ptr<A[], decltype(noop_deleter)> p1;
  p1.reset(arr, noop_deleter);
}

// Should NOT trigger for nullptr with reset
void test_nullptr_reset_ok() {
  std::unique_ptr<A[]> b;
  b.reset(nullptr);
}

// 
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
