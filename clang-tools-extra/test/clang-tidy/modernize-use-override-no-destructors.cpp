// RUN: %check_clang_tidy %s modernize-use-override %t -- \
// RUN:   -config="{CheckOptions: [{key: modernize-use-override.IgnoreDestructors, value: 1}]}"

struct Base {
  virtual ~Base();
  virtual void f();
};

struct Simple : public Base {
  virtual ~Simple();
  // CHECK-MESSAGES-NOT: warning:
  virtual void f();
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: prefer using
  // CHECK-FIXES: {{^}}  void f() override;
};
