// RUN: %check_clang_tidy %s modernize-use-override %t -- -- -fms-extensions -std=c++11

// This test is designed to test ms-extension __declspec(dllexport) attributes.
#define EXPORT __declspec(dllexport)

class Base {
  virtual EXPORT void a();
};

class EXPORT InheritedBase {
  virtual void a();
};

class Derived : public Base {
  virtual EXPORT void a();
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: prefer using 'override' or (rarely) 'final' instead of 'virtual' [modernize-use-override]
  // CHECK-FIXES: {{^}}  EXPORT void a() override;
};

class EXPORT InheritedDerived : public InheritedBase {
  virtual void a();
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: prefer using 'override' or (rarely) 'final' instead of 'virtual' [modernize-use-override]
  // CHECK-FIXES: {{^}}  void a() override;
};

