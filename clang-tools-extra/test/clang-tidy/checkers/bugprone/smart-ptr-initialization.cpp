// RUN: %check_clang_tidy -std=c++11-or-later %s bugprone-smart-ptr-initialization %t -- -- -I %S/../Inputs/Headers/std

#include <memory>
#include <utility>

struct A {
  int x;
};

A& getA();
A* getAPtr();

void test_shared_ptr_constructor() {
  std::shared_ptr<A> a(&getA());
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: passing a raw pointer 'A*' to 'std::shared_ptr<A>' constructor may cause double deletion
}

void test_unique_ptr_constructor() {
  std::unique_ptr<A> b(&getA());
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: passing a raw pointer 'A*' to 'std::unique_ptr<A>' constructor may cause double deletion
}

void test_stack_variable() {
  int x = 5;
  std::unique_ptr<int> ptr(&x);
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: passing a raw pointer 'int*' to 'std::unique_ptr<int>' constructor may cause double deletion
}

// Should trigger for member variables
struct S {
  int member;
  void test() {
    std::unique_ptr<int> ptr(&member);
    // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: passing a raw pointer 'int*' to 'std::unique_ptr<int>' constructor may cause double deletion
  }
};

void test_function_return() {
  std::shared_ptr<A> sp(getAPtr());
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: passing a raw pointer 'A*' to 'std::shared_ptr<A>' constructor may cause double deletion
}

void test_new_expression_ok() {
  std::shared_ptr<A> a(new A());
  std::unique_ptr<A> b(new A());
}

void test_release_ok(std::unique_ptr<A> p1, std::shared_ptr<A> p3) {
  std::unique_ptr<A> p2(p1.release());
}

void test_release_cast_ok(std::unique_ptr<A> p1, std::shared_ptr<A> p3) {
  std::unique_ptr<A> p2(static_cast<A*>(p1.release()));
}

struct NoopDeleter {
    void operator() (A* p) {}
};

void test_custom_deleter_ok() {
  auto noop_deleter = [](A* p) {  };
  std::unique_ptr<A, NoopDeleter> p0(&getA());
  std::unique_ptr<A, decltype(noop_deleter)> p1(&getA(), noop_deleter);
  std::shared_ptr<A> p2(&getA(), noop_deleter);
}

void test_nullptr_ok() {
  std::shared_ptr<A> a(nullptr);
  std::unique_ptr<A> b(nullptr);
}

void test_zero_ok() {
  std::shared_ptr<A> a(0);
  std::unique_ptr<A> b(0);
}

void test_copy_move_constructor_ok(std::shared_ptr<A> sp, std::unique_ptr<A> up) {
  auto sp2 = sp;

  auto sp3 = std::move(sp);
  auto up3 = std::move(up);
}

void test_shared_ptr_reset() {
  std::shared_ptr<A> a;
  a.reset(&getA());
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: passing a raw pointer 'A*' to 'std::shared_ptr<A>::reset' may cause double deletion
}

void test_unique_ptr_reset() {
  std::unique_ptr<A> b;
  b.reset(&getA());
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: passing a raw pointer 'A*' to 'std::unique_ptr<A>::reset' may cause double deletion
}

void test_stack_variable_reset() {
  int x = 5;
  std::unique_ptr<int> ptr;
  ptr.reset(&x);
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: passing a raw pointer 'int*' to 'std::unique_ptr<int>::reset' may cause double deletion
}

void test_function_return_reset() {
  std::shared_ptr<A> sp;
  sp.reset(getAPtr());
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: passing a raw pointer 'A*' to 'std::shared_ptr<A>::reset' may cause double deletion
}

void test_new_expression_reset_ok() {
  std::shared_ptr<A> a;
  a.reset(new A());
  std::unique_ptr<A> b;
  b.reset(new A());
}

void test_release_reset_ok(std::unique_ptr<A> p1, std::shared_ptr<A> p3) {
  std::unique_ptr<A> p2;
  p2.reset(p1.release());
}

void test_release_reset_cast_ok(std::unique_ptr<A> p1, std::shared_ptr<A> p3) {
  std::unique_ptr<A> p2;
  p2.reset(static_cast<A*>(p1.release()));
}

void test_custom_deleter_reset_ok() {
  auto noop_deleter = [](A* p) {  };
  std::unique_ptr<A, NoopDeleter> p0;
  p0.reset(&getA());
  std::unique_ptr<A, decltype(noop_deleter)> p1;
  p1.reset(&getA());
  std::shared_ptr<A> p2;
  // FIXME: mock shared_ptr must support reset with custom deleter
  // p2.reset(&getA(), noop_deleter);
}

void test_nullptr_reset_ok() {
  std::unique_ptr<A> b;
  b.reset(nullptr);
}

void test_zero_reset_ok() {
  std::unique_ptr<A> b;
  b.reset(0);
}

void test_reset_ok() {
  std::shared_ptr<A> a;
  a.reset();
  std::unique_ptr<A> b;
  b.reset();
}

// Edge case: should trigger for array new with wrong smart pointer
void test_array_new() {
  std::shared_ptr<A> sp(new A[10]); // This is actually wrong but not our check's concern
  sp.reset(new A[10]);
  // This would be caught by bugprone-shared-ptr-array-mismatch checks
}

template<typename T>
void test_shared_ptr_constructor_template() {
  T a(&getA());
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: passing a raw pointer 'A*' to 'std::shared_ptr<A>' constructor may cause double deletion
}

int a = (test_shared_ptr_constructor_template<std::shared_ptr<A>>(), 0);
