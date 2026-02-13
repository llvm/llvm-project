// RUN: %check_clang_tidy %s bugprone-smart-ptr-initialization %t -- -- -I%S

#include "Inputs/smart-ptr-initialization/std_smart_ptr.h"

struct A {
  int x;
};

A& getA();
A* getAPtr();

// TODO: std::shared_ptr<A[]> also must be supported

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

// TODO: all `reset` tests must be separately

// Should trigger the check for reset() method
void test_reset_method() {
  std::shared_ptr<A> sp;
  sp.reset(&getA());
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: passing a raw pointer '&getA()' to std::shared_ptr::reset() may cause double deletion [bugprone-smart-ptr-initialization]

  std::unique_ptr<A> up;
  up.reset(&getA());
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: passing a raw pointer '&getA()' to std::unique_ptr::reset() may cause double deletion [bugprone-smart-ptr-initialization]
}

// Should NOT trigger the check for reset() method with custom deleter
void test_reset_method_with_custom_deleter() {
  auto noop_deleter = [](A* p) {  };
  std::shared_ptr<A> sp(nullptr, noop_deleter);
  std::unique_ptr<A, decltype(noop_deleter)> up(nullptr, noop_deleter);

  sp.reset(&getA());
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: passing a raw pointer '&getA()' to std::shared_ptr::reset() may cause double deletion [bugprone-smart-ptr-initialization]
  // doesn't have deleter anymore

  sp.reset(&getA(), noop_deleter);

  up.reset(&getA());
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
  // TODO: forbid to pass `new A[];`??
  std::shared_ptr<A> a(new A());
  std::unique_ptr<A> b(new A());
}

// Should NOT trigger for release() calls - ownership transfer
void test_release_ok() {
  auto p1 = std::make_unique<A>();
  std::unique_ptr<A> p2(p1.release());
  
  auto p3 = std::make_shared<A>();
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
  std::shared_ptr<A> c;
  c.reset(nullptr);
}

// Should NOT trigger for make_shared/make_unique
void test_make_functions_ok() {
  auto sp = std::make_shared<A>();
  auto up = std::make_unique<A>();
}

// TODO: write that this is a job for bugprone-shared-ptr-array-mismatch
// TODO: the same test, but for smart pointer of array
// TODO: the same test, but with `release` call
// 
// Edge case: should trigger for array new with wrong smart pointer
void test_array_new() {
  std::shared_ptr<A> sp(new A[10]); // This is actually wrong but not our check's concern
  // This would be caught by other checks (mismatched new/delete)
}
