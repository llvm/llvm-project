// RUN: %check_clang_tidy -std=c++11-or-later %s bugprone-smart-ptr-initialization %t

#include <memory>

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

void test_new_expression_cast_ok() {
  A* first = static_cast<A*>(new A());
  A* second = static_cast<A*>(new A());
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


// test_new_expression_ok_in_global
// FIXME: support it
/*
A* first = new A();
A* second = new A();
std::shared_ptr<A> a(first);
std::unique_ptr<A> b(second);

// test_new_expression_fail_in_global
A* first2 = new A();
A* second2 = new A();
std::shared_ptr<A> a1(first2);
std::shared_ptr<A> a2(first2);
std::unique_ptr<A> b1(second2);
std::unique_ptr<A> b2(second2);
*/


void test_new_expression_reset_ok() {
  A* first = new A();
  A* second = new A();
  std::shared_ptr<A> a;
  a.reset(first);
  std::unique_ptr<A> b;
  b.reset(second);
}

void test_new_expression_reset_fail() {
  A* first = new A();
  A* second = new A();
  std::shared_ptr<A> a;
  a.reset(first);
  std::shared_ptr<A> a2;
  a2.reset(first);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: passing a raw pointer 'A*' to 'std::shared_ptr<A>::reset' may cause double deletion
  std::unique_ptr<A> b;
  b.reset(second);
  std::unique_ptr<A> b2;
  b2.reset(second);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: passing a raw pointer 'A*' to 'std::unique_ptr<A>::reset' may cause double deletion
}

void test_new_expression_crossed_fail() {
  A* first = new A();
  A* second = new A();
  std::shared_ptr<A> a(first);
  std::shared_ptr<A> a2;
  a2.reset(first);
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: passing a raw pointer 'A*' to 'std::shared_ptr<A>::reset' may cause double deletion
  std::unique_ptr<A> b;
  b.reset(second);
  std::unique_ptr<A> b2(second);
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: passing a raw pointer 'A*' to 'std::unique_ptr<A>' constructor may cause double deletion
}

bool can_take(std::shared_ptr<A> a);
void take(std::shared_ptr<A> a);

void test_discowered_in_wild(std::shared_ptr<A> a, std::unique_ptr<A> b) {
  if (can_take(a)) {
    take(a);
  }
}
