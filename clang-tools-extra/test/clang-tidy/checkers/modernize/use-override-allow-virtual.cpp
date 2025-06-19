// RUN: %check_clang_tidy %s modernize-use-override %t -- \
// RUN:   -config="{CheckOptions: {modernize-use-override.AllowVirtual: true}}"

struct Base {
  virtual ~Base();
  virtual void a();
  virtual void b();
  virtual void c();
};

struct Derived : public Base {
  virtual ~Derived() override;

  virtual void a() override;
  // CHECK-MESSAGES-NOT: warning:
  // CHECK-FIXES: {{^}}  virtual void a() override;

  virtual void b();
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: add 'override'
  // CHECK-FIXES: {{^}}  virtual void b() override;

  void c();
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: annotate
  // CHECK-FIXES: {{^}}  void c() override;
};
