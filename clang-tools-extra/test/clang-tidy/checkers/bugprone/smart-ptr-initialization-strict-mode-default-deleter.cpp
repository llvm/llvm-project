// RUN: %check_clang_tidy -std=c++11-or-later %s bugprone-smart-ptr-initialization %t -- -config="{CheckOptions: {bugprone-smart-ptr-initialization.StrictMode: 'true'}}"

#include <memory>

struct A {
  int x;
};

A& getA();
A* getAPtr();

void test_shared_ptr_constructor() {
  std::shared_ptr<A> a(&getA(), std::default_delete<A>{});
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: passing a raw pointer 'A *' to 'std::shared_ptr<A>' constructor may cause double deletion
}

void test_unique_ptr_constructor() {
  std::unique_ptr<A, std::default_delete<A>> b(&getA());
  // CHECK-MESSAGES: :[[@LINE-1]]:48: warning: passing a raw pointer 'A *' to 'std::unique_ptr<A, std::default_delete<A>>' constructor may cause double deletion
}

void test_stack_variable() {
  int x = 5;
  std::unique_ptr<int, std::default_delete<int>> ptr(&x);
  // CHECK-MESSAGES: :[[@LINE-1]]:54: warning: passing a raw pointer 'int *' to 'std::unique_ptr<int, std::default_delete<int>>' constructor may cause double deletion
}

// Should trigger for member variables
struct S {
  int member;
  void test() {
    std::unique_ptr<int, std::default_delete<int>> ptr(&member);
    // CHECK-MESSAGES: :[[@LINE-1]]:56: warning: passing a raw pointer 'int *' to 'std::unique_ptr<int, std::default_delete<int>>' constructor may cause double deletion
  }
};

void test_function_return() {
  std::shared_ptr<A> sp(getAPtr(), std::default_delete<A>{});
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: passing a raw pointer 'A *' to 'std::shared_ptr<A>' constructor may cause double deletion
}

void test_new_expression_ok() {
  std::shared_ptr<A> a(new A(), std::default_delete<A>{});
  std::unique_ptr<A, std::default_delete<A>> b(new A());
}

void test_release_ok(std::unique_ptr<A> p1, std::shared_ptr<A> p3) {
  std::unique_ptr<A, std::default_delete<A>> p2(p1.release());
}

void test_release_cast_ok(std::unique_ptr<A> p1, std::shared_ptr<A> p3) {
  std::unique_ptr<A, std::default_delete<A>> p2(static_cast<A*>(p1.release()));
}

void test_nullptr_ok() {
  // FIXME: mock shared_ptr must support it
  // std::shared_ptr<A> a(nullptr, std::default_delete<A>{});
  std::unique_ptr<A, std::default_delete<A>> b(nullptr);
}

void test_zero_ok() {
  // FIXME: mock shared_ptr must support it
  // std::shared_ptr<A> a(0, std::default_delete<A>{});
  std::unique_ptr<A, std::default_delete<A>> b(0, std::default_delete<A>{});
}

void test_shared_ptr_reset() {
  std::shared_ptr<A> a;
  a.reset(&getA(), std::default_delete<A>{});
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: passing a raw pointer 'A *' to 'std::shared_ptr<A>' reset method may cause double deletion
}

void test_unique_ptr_reset() {
  std::unique_ptr<A, std::default_delete<A>> b;
  b.reset(&getA());
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: passing a raw pointer 'A *' to 'std::unique_ptr<A, std::default_delete<A>>' reset method may cause double deletion
}

void test_stack_variable_reset() {
  int x = 5;
  std::unique_ptr<int, std::default_delete<A>> ptr;
  ptr.reset(&x);
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: passing a raw pointer 'int *' to 'std::unique_ptr<int, std::default_delete<A>>' reset method may cause double deletion
}

void test_function_return_reset() {
  std::shared_ptr<A> sp;
  sp.reset(getAPtr(), std::default_delete<A>{});
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: passing a raw pointer 'A *' to 'std::shared_ptr<A>' reset method may cause double deletion
}

void test_new_expression_reset_ok() {
  std::shared_ptr<A> a;
  a.reset(new A(), std::default_delete<A>{});
  std::unique_ptr<A, std::default_delete<A>> b;
  b.reset(new A());
}

void test_release_reset_ok(std::unique_ptr<A> p1) {
  std::unique_ptr<A, std::default_delete<A>> p2;
  p2.reset(p1.release());
}

void test_release_reset_cast_ok(std::unique_ptr<A> p1) {
  std::unique_ptr<A, std::default_delete<A>> p2;
  p2.reset(static_cast<A*>(p1.release()));
}

void test_nullptr_reset_ok() {
  std::unique_ptr<A, std::default_delete<A>> b;
  b.reset(nullptr);
}

void test_zero_reset_ok() {
  std::unique_ptr<A, std::default_delete<A>> b;
  b.reset(0);
}

void test_reset_ok() {
  std::unique_ptr<A, std::default_delete<A>> b;
  b.reset();
}

