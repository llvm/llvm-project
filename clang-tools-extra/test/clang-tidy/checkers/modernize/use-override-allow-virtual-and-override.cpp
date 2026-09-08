// RUN: %check_clang_tidy -check-suffix=VO %s modernize-use-override %t -- \
// RUN:   -config="{CheckOptions: {modernize-use-override.AllowVirtualAndOverride: true}}"
// RUN: %check_clang_tidy -check-suffix=VOF %s modernize-use-override %t -- \
// RUN:   -config="{CheckOptions: {modernize-use-override.AllowVirtualAndOverride: true, \
// RUN:                            modernize-use-override.AllowOverrideAndFinal: true}}"

struct Base {
  virtual void a();
  virtual void b();
  virtual void c();
  virtual void d();
  virtual void e();
  virtual void f();
};

struct Derived : public Base {
  virtual void a() override;

  void b() override;

  virtual void c();
  // CHECK-MESSAGES-VO: :[[@LINE-1]]:16: warning: prefer using 'override' or (rarely) 'final' instead of 'virtual'
  // CHECK-MESSAGES-VOF: :[[@LINE-2]]:16: warning: prefer using 'override' or (rarely) 'final' instead of 'virtual'
  // CHECK-FIXES-VO: void c() override;
  // CHECK-FIXES-VOF: void c() override;

  void d();
  // CHECK-MESSAGES-VO: :[[@LINE-1]]:8: warning: annotate this function with 'override' or (rarely) 'final'
  // CHECK-MESSAGES-VOF: :[[@LINE-2]]:8: warning: annotate this function with 'override' or (rarely) 'final'
  // CHECK-FIXES-VO: void d() override;
  // CHECK-FIXES-VOF: void d() override;

  virtual void e() final;
  // CHECK-MESSAGES-VO: :[[@LINE-1]]:16: warning: 'virtual' is redundant since the function is already declared 'final'
  // CHECK-MESSAGES-VOF: :[[@LINE-2]]:16: warning: 'virtual' is redundant since the function is already declared 'final'
  // CHECK-FIXES-VO: void e() final;
  // CHECK-FIXES-VOF: void e() final;

  virtual void f() override final;
  // CHECK-MESSAGES-VO: :[[@LINE-1]]:16: warning: 'virtual' and 'override' are redundant since the function is already declared 'final'
  // CHECK-FIXES-VO: void f() final;
};
