// RUN: %check_clang_tidy -std=c++17-or-later %s bugprone-smart-ptr-initialization %t -- -- -I%S

#include "Inputs/smart-ptr-initialization/std_smart_ptr.h"

struct A {
  int x;
};

A arr[10];

// Should trigger the check for shared_ptr constructor
void test_shared_ptr_constructor() {
  std::shared_ptr<A[]> a(arr);
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: passing a raw pointer 'arr' to std::shared_ptr constructor may cause double deletion [bugprone-smart-ptr-initialization]
}

// Should trigger for stack variables
void test_stack_variable() {
  int x[10] = {5};
  std::shared_ptr<int[]> ptr(x);
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: passing a raw pointer 'x' to std::shared_ptr constructor may cause double deletion [bugprone-smart-ptr-initialization]
}

// Should trigger for member variables
struct S {
  int member[10];
  void test() {
    std::shared_ptr<int[]> ptr(member);
    // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: passing a raw pointer 'this->member' to std::shared_ptr constructor may cause double deletion [bugprone-smart-ptr-initialization]
  }
};

// Should NOT trigger for new expressions - these are OK
void test_new_expression_ok() {
  std::shared_ptr<A[]> a(new A[10]);
}

// Should NOT trigger for release() calls - ownership transfer
void test_release_ok(std::shared_ptr<A[]> p3) {
  std::shared_ptr<A[]> p4(p3.release());
}

struct NoopDeleter {
    void operator() (A* p) {}
};

// Should NOT trigger for custom deleters
void test_custom_deleter_ok() {
  auto noop_deleter = [](A* p) {  };
  std::shared_ptr<A[]> p2(arr, noop_deleter);
}

// Should NOT trigger for nullptr
void test_nullptr_ok() {
  std::shared_ptr<A[]> a(nullptr);
}

// Should NOT trigger for copy and move constructors
void test_copy_move_constructor_ok(std::shared_ptr<A[]> sp) {
  auto sp2 = sp;
  auto sp3 = std::move(sp);
}

// Should trigger the check for shared_ptr reset
void test_shared_ptr_reset() {
  std::shared_ptr<A[]> a;
  a.reset(arr);
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: passing a raw pointer 'arr' to std::shared_ptr::reset() may cause double deletion [bugprone-smart-ptr-initialization]
}

// Should trigger for stack variables with reset
void test_stack_variable_reset() {
  int x[10] = {5};
  std::shared_ptr<int[]> ptr;
  ptr.reset(x);
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: passing a raw pointer 'x' to std::shared_ptr::reset() may cause double deletion [bugprone-smart-ptr-initialization]
}

// Should NOT trigger for new expressions with reset - these are OK
void test_new_expression_reset_ok() {
  std::shared_ptr<A[]> a;
  a.reset(new A[10]);
}

// Should NOT trigger for release() calls with reset - ownership transfer
void test_release_reset_ok(std::shared_ptr<A[]> p3) {
  std::shared_ptr<A[]> p4;
  p4.reset(p3.release());
}

// Should NOT trigger for custom deleters with reset
void test_custom_deleter_reset_ok() {
  auto noop_deleter = [](A* p) {  };
  std::shared_ptr<A[]> p2;
  p2.reset(arr, noop_deleter);
}

// Should NOT trigger for nullptr with reset
void test_nullptr_reset_ok() {
  std::shared_ptr<A[]> a;
  a.reset(nullptr);
}

// 
// Edge case: should trigger for array new with wrong smart pointer
void test_array_new() {
  std::shared_ptr<A[]> sp(new A); // This is actually wrong but not our check's concern
  sp.reset(new A);
  // This would be caught by bugprone-shared-ptr-array-mismatch checks
}

void test_array_release(std::shared_ptr<A> spa) {
  std::shared_ptr<A[]> sp(spa.release()); // This is actually wrong but not our check's concern
  sp.reset(spa.release());
  // This would be caught by bugprone-shared-ptr-array-mismatch checks (mismatched new/delete)
}
