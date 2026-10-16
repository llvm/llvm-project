// RUN: %check_clang_tidy %s modernize-use-override %t -- \
// RUN:   -config="{CheckOptions: {modernize-use-override.OverrideSpelling: 'CUSTOM_OVERRIDE',modernize-use-override.FinalSpelling: 'CUSTOM_FINAL'}}"

// As if the macro was not defined.
//#define CUSTOM_OVERRIDE override
//#define CUSTOM_FINAL override

struct Base {
  virtual ~Base() {}
  virtual void a();
  virtual void b();
};

struct SimpleCases : public Base {
public:
  virtual ~SimpleCases();
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: prefer using 'CUSTOM_OVERRIDE' or (rarely) 'CUSTOM_FINAL' instead of 'virtual' [modernize-use-override]
  // CHECK-FIXES: virtual ~SimpleCases();

  void a();
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: annotate this function with 'CUSTOM_OVERRIDE' or (rarely) 'CUSTOM_FINAL' [modernize-use-override]
  // CHECK-FIXES: void a();

  virtual void b();
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: prefer using 'CUSTOM_OVERRIDE' or (rarely) 'CUSTOM_FINAL' instead of 'virtual' [modernize-use-override]
  // CHECK-FIXES: virtual void b();
};
