// RUN: %check_clang_tidy -std=c++11-or-later %s bugprone-smart-ptr-initialization %t -- -- -I%S

#include "Inputs/smart-ptr-initialization/std_smart_ptr.h"

struct A {
  int x;
};

A& getA();
A* getAPtr();

// Should trigger the check for shared_ptr constructor
void test_shared_ptr_constructor() {
  std::shared_ptr<A> a(&getA());
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: passing a raw pointer '&getA()' to std::shared_ptr constructor may cause double deletion [bugprone-smart-ptr-initialization]
}

// Should trigger the check for unique_ptr constructor  
void test_unique_ptr_constructor() {
  std::unique_ptr<A> b(&getA());
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: passing a raw pointer '&getA()' to std::unique_ptr constructor may cause double deletion [bugprone-smart-ptr-initialization]
}

// Should trigger for stack variables
void test_stack_variable() {
  int x = 5;
  std::unique_ptr<int> ptr(&x);
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: passing a raw pointer '&x' to std::unique_ptr constructor may cause double deletion [bugprone-smart-ptr-initialization]
}

// Should trigger for member variables
struct S {
  int member;
  void test() {
    std::unique_ptr<int> ptr(&member);
    // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: passing a raw pointer '&this->member' to std::unique_ptr constructor may cause double deletion [bugprone-smart-ptr-initialization]
  }
};

// Should trigger for pointer returned from function
void test_function_return() {
  std::shared_ptr<A> sp(getAPtr());
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: passing a raw pointer 'getAPtr()' to std::shared_ptr constructor may cause double deletion [bugprone-smart-ptr-initialization]
}

// Should NOT trigger for new expressions - these are OK
void test_new_expression_ok() {
  std::shared_ptr<A> a(new A());
  std::unique_ptr<A> b(new A());
}

// Should NOT trigger for release() calls - ownership transfer
void test_release_ok(std::unique_ptr<A> p1, std::shared_ptr<A> p3) {
  std::unique_ptr<A> p2(p1.release());
  std::shared_ptr<A> p4(p3.release());
}

struct NoopDeleter {
    void operator() (A* p) {}
};

// Should NOT trigger for custom deleters
void test_custom_deleter_ok() {
  auto noop_deleter = [](A* p) {  };
  std::unique_ptr<A, NoopDeleter> p0(&getA());
  std::unique_ptr<A, decltype(noop_deleter)> p1(&getA(), noop_deleter);
  std::shared_ptr<A> p2(&getA(), noop_deleter);
}

// Should NOT trigger for nullptr
void test_nullptr_ok() {
  std::shared_ptr<A> a(nullptr);
  std::unique_ptr<A> b(nullptr);
}

// Should NOT trigger for copy and move constructors
void test_copy_move_constructor_ok(std::shared_ptr<A> sp, std::unique_ptr<A> up) {
  auto sp2 = sp;

  auto sp3 = std::move(sp);
  auto up3 = std::move(up);
}

// Should trigger the check for shared_ptr reset
void test_shared_ptr_reset() {
  std::shared_ptr<A> a;
  a.reset(&getA());
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: passing a raw pointer '&getA()' to std::shared_ptr::reset() may cause double deletion [bugprone-smart-ptr-initialization]
}

// Should trigger the check for unique_ptr reset
void test_unique_ptr_reset() {
  std::unique_ptr<A> b;
  b.reset(&getA());
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: passing a raw pointer '&getA()' to std::unique_ptr::reset() may cause double deletion [bugprone-smart-ptr-initialization]
}

// Should trigger for stack variables with reset
void test_stack_variable_reset() {
  int x = 5;
  std::unique_ptr<int> ptr;
  ptr.reset(&x);
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: passing a raw pointer '&x' to std::unique_ptr::reset() may cause double deletion [bugprone-smart-ptr-initialization]
}

// Should trigger for pointer returned from function with reset
void test_function_return_reset() {
  std::shared_ptr<A> sp;
  sp.reset(getAPtr());
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: passing a raw pointer 'getAPtr()' to std::shared_ptr::reset() may cause double deletion [bugprone-smart-ptr-initialization]
}

// Should NOT trigger for new expressions with reset - these are OK
void test_new_expression_reset_ok() {
  std::shared_ptr<A> a;
  a.reset(new A());
  std::unique_ptr<A> b;
  b.reset(new A());
}

// Should NOT trigger for release() calls with reset - ownership transfer
void test_release_reset_ok(std::unique_ptr<A> p1, std::shared_ptr<A> p3) {
  std::unique_ptr<A> p2;
  p2.reset(p1.release());
  std::shared_ptr<A> p4;
  p4.reset(p3.release());
}

// Should NOT trigger for custom deleters with reset
void test_custom_deleter_reset_ok() {
  auto noop_deleter = [](A* p) {  };
  std::unique_ptr<A, NoopDeleter> p0;
  p0.reset(&getA());
  std::unique_ptr<A, decltype(noop_deleter)> p1;
  p1.reset(&getA(), noop_deleter);
  std::shared_ptr<A> p2;
  p2.reset(&getA(), noop_deleter);
}

// Should NOT trigger for nullptr with reset
void test_nullptr_reset_ok() {
  std::shared_ptr<A> a;
  a.reset(nullptr);
  std::unique_ptr<A> b;
  b.reset(nullptr);
}

// 
// Edge case: should trigger for array new with wrong smart pointer
void test_array_new() {
  std::shared_ptr<A> sp(new A[10]); // This is actually wrong but not our check's concern
  sp.reset(new A[10]);
  // This would be caught by bugprone-shared-ptr-array-mismatch checks
}

void test_array_release(std::shared_ptr<A[]> spa) {
  std::shared_ptr<A> sp(spa.release()); // This is actually wrong but not our check's concern
  sp.reset(spa.release());
  // This would be caught by bugprone-shared-ptr-array-mismatch checks (mismatched new/delete)
}
