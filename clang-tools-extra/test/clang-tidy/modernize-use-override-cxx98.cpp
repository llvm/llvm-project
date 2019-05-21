// RUN: %check_clang_tidy -std=c++98 %s modernize-use-override %t

struct Base {
  virtual ~Base() {}
  virtual void a();
  virtual void b();
};

struct SimpleCases : public Base {
public:
  virtual ~SimpleCases();
  // CHECK-FIXES: {{^}}  virtual ~SimpleCases();

  void a();
  // CHECK-FIXES: {{^}}  void a();

  virtual void b();
  // CHECK-FIXES: {{^}}  virtual void b();
};
