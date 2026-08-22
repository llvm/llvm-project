// RUN: %check_clang_tidy -std=c++11-or-later %s bugprone-smart-ptr-initialization %t -- -- -I %S/../Inputs/Headers/std

#include <memory>
#include <utility>

struct A {
  int x;
};

A* getAPtr();

void test_new_expression_ok() {
  A* first = new A();
  A* second = new A();
  std::shared_ptr<A> a(first);
  std::unique_ptr<A> b(second);
}

void test_new_expression_fail() {
  A* first = new A();
  A* second = new A();
  std::shared_ptr<A> a(first);
  std::shared_ptr<A> a2(first);
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: passing a raw pointer 'A*' to 'std::shared_ptr<A>' constructor may cause double deletion
  std::unique_ptr<A> b(second);
  std::unique_ptr<A> b2(second);
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: passing a raw pointer 'A*' to 'std::unique_ptr<A>' constructor may cause double deletion
}

void test_new_expression_fail_in_different_scopes() {
  A* first = new A();
  A* second = new A();
  {
  std::shared_ptr<A> a(first);
  }
  {
  std::shared_ptr<A> a2(first);
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: passing a raw pointer 'A*' to 'std::shared_ptr<A>' constructor may cause double deletion
  }
  {
  std::unique_ptr<A> b(second);
  }
  {
  std::unique_ptr<A> b2(second);
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: passing a raw pointer 'A*' to 'std::unique_ptr<A>' constructor may cause double deletion
  }
}

void test_new_expression_ok_in_scope() {
  {
  A* first = new A();
  A* second = new A();
  std::shared_ptr<A> a(first);
  std::unique_ptr<A> b(second);
  }
}

void test_new_expression_fail_in_scope() {
  {
  A* first = new A();
  A* second = new A();
  std::shared_ptr<A> a(first);
  std::shared_ptr<A> a2(first);
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: passing a raw pointer 'A*' to 'std::shared_ptr<A>' constructor may cause double deletion
  std::unique_ptr<A> b(second);
  std::unique_ptr<A> b2(second);
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: passing a raw pointer 'A*' to 'std::unique_ptr<A>' constructor may cause double deletion
  }
}

struct B {

void test_new_expression_ok_as_method() {
  A* first = new A();
  A* second = new A();
  std::shared_ptr<A> a(first);
  std::unique_ptr<A> b(second);
}

void test_new_expression_fail_as_method() {
  A* first = new A();
  A* second = new A();
  std::shared_ptr<A> a(first);
  std::shared_ptr<A> a2(first);
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: passing a raw pointer 'A*' to 'std::shared_ptr<A>' constructor may cause double deletion
  std::unique_ptr<A> b(second);
  std::unique_ptr<A> b2(second);
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: passing a raw pointer 'A*' to 'std::unique_ptr<A>' constructor may cause double deletion
}

};

// TODO: reset 