// RUN: %check_clang_tidy %s modernize-use-override %t -- \
// RUN:   -config="{CheckOptions: [{key: modernize-use-override.AllowOverrideAndFinal, value: true}]}"

struct Base {
  virtual ~Base();
  virtual void a();
  virtual void b();
  virtual void c();
  virtual void d();
  virtual void e();
  virtual void f();
  virtual void g();
  virtual void h();
  virtual void i();
};

struct Simple : public Base {
  virtual ~Simple();
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: prefer using 'override' or (rarely) 'final' instead of 'virtual' [modernize-use-override]
  // CHECK-FIXES: {{^}}  ~Simple() override;
  virtual void a() override;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: 'virtual' is redundant since the function is already declared 'override' [modernize-use-override]
  // CHECK-FIXES: {{^}}  void a() override;
  virtual void b() final;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: 'virtual' is redundant since the function is already declared 'final' [modernize-use-override]
  // CHECK-FIXES: {{^}}  void b() final;
  virtual void c() final override;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: 'virtual' is redundant since the function is already declared 'final' [modernize-use-override]
  // CHECK-FIXES: {{^}}  void c() final override;
  virtual void d() override final;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: 'virtual' is redundant since the function is already declared 'final' [modernize-use-override]
  // CHECK-FIXES: {{^}}  void d() override final;
  void e() final override;
  void f() override final;
  void g() final;
  void h() override;
  void i();
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: annotate this function with 'override' or (rarely) 'final' [modernize-use-override]
  // CHECK-FIXES: {{^}}  void i() override;
};
