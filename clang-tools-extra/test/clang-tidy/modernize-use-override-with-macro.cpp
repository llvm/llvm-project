// RUN: %check_clang_tidy %s modernize-use-override %t -- \
// RUN:   -config="{CheckOptions: [{key: modernize-use-override.OverrideSpelling, value: 'OVERRIDE'},{key: modernize-use-override.FinalSpelling, value: 'FINAL'}]}" \
// RUN: -- -std=c++11

#define ABSTRACT = 0

#define OVERRIDE override
#define FINAL final
#define VIRTUAL virtual
#define NOT_VIRTUAL
#define NOT_OVERRIDE

#define MUST_USE_RESULT __attribute__((warn_unused_result))
#define UNUSED __attribute__((unused))

struct Base {
  virtual ~Base() {}
  virtual void a();
  virtual void b();
  virtual void c();
  virtual void e() = 0;
  virtual void f2() const = 0;
  virtual void g() = 0;
  virtual void j() const;
  virtual void k() = 0;
  virtual void l() const;
};

struct SimpleCases : public Base {
public:
  virtual ~SimpleCases();
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: prefer using 'OVERRIDE' or (rarely) 'FINAL' instead of 'virtual' [modernize-use-override]
  // CHECK-FIXES: {{^}}  ~SimpleCases() OVERRIDE;

  void a();
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: annotate this function with 'OVERRIDE' or (rarely) 'FINAL' [modernize-use-override]
  // CHECK-FIXES: {{^}}  void a() OVERRIDE;

  virtual void b();
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: prefer using 'OVERRIDE' or (rarely) 'FINAL' instead of 'virtual' [modernize-use-override]
  // CHECK-FIXES: {{^}}  void b() OVERRIDE;

  virtual void c();
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: prefer using
  // CHECK-FIXES: {{^}}  void c() OVERRIDE;

  virtual void e() = 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: prefer using
  // CHECK-FIXES: {{^}}  void e() OVERRIDE = 0;

  virtual void f2() const = 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: prefer using
  // CHECK-FIXES: {{^}}  void f2() const OVERRIDE = 0;

  virtual void g() ABSTRACT;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: prefer using
  // CHECK-FIXES: {{^}}  void g() OVERRIDE ABSTRACT;

  virtual void j() const;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: prefer using
  // CHECK-FIXES: {{^}}  void j() const OVERRIDE;

  virtual void k() OVERRIDE;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: 'virtual' is redundant since the function is already declared 'OVERRIDE' [modernize-use-override]
  // CHECK-FIXES: {{^}}  void k() OVERRIDE;

  virtual void l() const OVERRIDE;
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: 'virtual' is redundant since the function is already declared 'OVERRIDE' [modernize-use-override]
  // CHECK-FIXES: {{^}}  void l() const OVERRIDE;
};
