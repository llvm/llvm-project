// RUN: %check_clang_tidy -std=c++26-or-later %s modernize-use-override,cppcoreguidelines-explicit-virtual-functions %t

struct Base {
  virtual void f() = delete("");
};

struct Derived : Base {
  virtual void f() = delete("");
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: prefer using 'override' or (rarely) 'final' instead of 'virtual'
  // CHECK-FIXES: void f() override = delete("");
};
