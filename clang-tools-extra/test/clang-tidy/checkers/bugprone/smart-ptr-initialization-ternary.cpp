// RUN: %check_clang_tidy -std=c++11-or-later %s bugprone-smart-ptr-initialization %t -- -- -I %S/../Inputs/Headers/std

#include <memory>

struct A {
  int x;
};

A& getA();
A* getAPtr();
bool flag = false;

void test_new_expression_ok() {
  std::shared_ptr<A> a(flag ? new A() : nullptr);
  std::unique_ptr<A> b(flag ? nullptr : new A());
}

void test_release_ok(std::unique_ptr<A> p1, std::shared_ptr<A> p3) {
  std::unique_ptr<A> p2(flag ? p1.release() : nullptr);
}

void test_new_expression_reset_ok() {
  std::shared_ptr<A> a;
  a.reset(flag ? new A() : nullptr);
  std::unique_ptr<A> b;
  b.reset(flag ? nullptr : new A());
}

void test_release_reset_ok(std::unique_ptr<A> p1, std::shared_ptr<A> p3) {
  std::unique_ptr<A> p2;
  p2.reset(flag ? p1.release() : nullptr);
}
