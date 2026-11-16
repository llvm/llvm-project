// RUN: %check_clang_tidy -std=c++98 %s readability-redundant-override %t
// RUN: %clang_cc1 -std=c++98 -fsyntax-only %t.cpp

struct Base {
  virtual void f() {}
};
struct Derived : Base {
  virtual void f() { Base::f(); }
};
// CHECK-MESSAGES: :[[@LINE-2]]:16: warning: redundant override
// CHECK-FIXES: struct Derived : Base {
// CHECK-FIXES-NOT: f()
// CHECK-FIXES: };
