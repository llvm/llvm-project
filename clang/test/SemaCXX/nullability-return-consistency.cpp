// Redeclarations must agree on return nullability, as they already must for
// parameters. An override may not weaken its base's contract: a _Nullable
// return for a _Nonnull one, or a _Nonnull parameter for a _Nullable one.
// Strengthening either way is a valid substitution and stays silent.

// RUN: %clang_cc1 -fsyntax-only %s -verify

int *_Nonnull f(); // expected-note {{previous declaration is here}}
int *_Nullable f(); // expected-warning {{nullability specifier '_Nullable' conflicts with existing specifier '_Nonnull'}}

int *_Nonnull same();
int *_Nonnull same();

int *_Nonnull left_unannotated();
int *left_unannotated();

struct B {
  virtual int *_Nonnull ret(); // expected-note {{overridden virtual function is here}}
  virtual int *_Nullable ret_ok();
  virtual void param(int *_Nullable p); // expected-note {{overridden virtual function is here}}
  virtual void param_ok(int *_Nonnull p);
  virtual int *unannotated(int *p);
};

struct D : B {
  int *_Nullable ret() override; // expected-warning {{conflicting nullability specifier on return types, '_Nullable' conflicts with existing specifier '_Nonnull'}}
  int *_Nonnull ret_ok() override;
  void param(int *_Nonnull p) override; // expected-warning {{conflicting nullability specifier on parameter types, '_Nonnull' conflicts with existing specifier '_Nullable'}}
  void param_ok(int *_Nullable p) override;
  int *_Nullable unannotated(int *_Nonnull p) override;
};
