// RUN: %clang_cc1 %s -verify -fsyntax-only

// expected-no-diagnostics

class C_in_class {
#include "../Sema/attr-callback.c"
};

struct Base {

  void no_args_1(void (*callback)(void));
  __attribute__((callback(1))) void no_args_2(void (*callback)(void));
  __attribute__((callback(callback))) void no_args_3(void (*callback)(void)) {}

  __attribute__((callback(1, 0))) virtual void
  this_tr(void (*callback)(Base *));

  __attribute__((callback(1, this, __, this))) virtual void
  this_unknown_this(void (*callback)(Base *, Base *, Base *));

  __attribute__((callback(1))) virtual void
  virtual_1(void (*callback)(void));

  __attribute__((callback(callback))) virtual void
  virtual_2(void (*callback)(void));

  __attribute__((callback(1))) virtual void
  virtual_3(void (*callback)(void));
};

__attribute__((callback(1))) void
Base::no_args_1(void (*callback)(void)) {
}

void Base::no_args_2(void (*callback)(void)) {
}

struct Derived_1 : public Base {

  __attribute__((callback(1, 0))) virtual void
  this_tr(void (*callback)(Base *)) override;

  __attribute__((callback(1))) virtual void
  virtual_1(void (*callback)(void)) override {}

  virtual void
  virtual_3(void (*callback)(void)) override {}
};

struct Derived_2 : public Base {

  __attribute__((callback(callback))) virtual void
  virtual_1(void (*callback)(void)) override;

  virtual void
  virtual_2(void (*callback)(void)) override;

  virtual void
  virtual_3(void (*callback)(void)) override;
};

void Derived_2::virtual_1(void (*callback)(void)) {}

__attribute__((callback(1))) void
Derived_2::virtual_2(void (*callback)(void)) {}

void Derived_2::virtual_3(void (*callback)(void)) {}
