// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// expected-no-diagnostics

struct Base {
  virtual void override();
};

struct S : Base {
  virtual void final() final;
  virtual void override() override;
};

struct T : Base {
  virtual void override() override { } 
};

struct override;
struct Base2 {
  virtual override override(int override);
};

struct A : Base2 {
  virtual struct override override(int override) override;
};
