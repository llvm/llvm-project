// RUN: %clang_cc1 -fsyntax-only -Wlifetime-safety-overridden-missing-lifetimebound -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wlifetime-safety-annotation-placement -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wlifetime-safety-all -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected-none %s
// expected-none-no-diagnostics

struct BaseNamed {
  virtual const int &f(const int &x); // expected-note {{overridden method parameter is here}}
};

struct DerivedNamed : BaseNamed {
  const int &f(const int &x [[clang::lifetimebound]]) override; // expected-warning {{overriding method has 'lifetimebound' attribute on parameter 'x', but overridden method is not marked 'lifetimebound'}}
};

struct BaseUnnamed {
  virtual const int &f(const int &); // expected-note {{overridden method parameter is here}}
};

struct DerivedUnnamed : BaseUnnamed {
  const int &f([[clang::lifetimebound]] const int &) override; // expected-warning {{overriding method has 'lifetimebound' attribute on an unnamed parameter, but overridden method is not marked 'lifetimebound'}}
};

struct BaseThis {
  virtual const int &f() const; // expected-note {{overridden method is here}}
};

struct DerivedThis : BaseThis {
  const int &f() const [[clang::lifetimebound]] override; // expected-warning {{overriding method has 'lifetimebound' attribute on the implicit object parameter, but overridden method is not marked 'lifetimebound'}}
};

struct BaseBoth {
  virtual const int &f(const int &x [[clang::lifetimebound]]) const [[clang::lifetimebound]];
};

struct DerivedBoth : BaseBoth {
  const int &f(const int &x [[clang::lifetimebound]]) const [[clang::lifetimebound]] override; // no warning
};

struct BaseWeaker {
  virtual const int &f(const int &x [[clang::lifetimebound]]) const [[clang::lifetimebound]];
};

struct DerivedWeaker : BaseWeaker {
  const int &f(const int &x) const override; // no warning under LSP
};

struct GrandBase {
  virtual const int &f(const int &x);
};

struct Middle : GrandBase {
  const int &f(const int &x) override; // expected-note {{overridden method parameter is here}}
};

struct Leaf : Middle {
  const int &f(const int &x [[clang::lifetimebound]]) override; // expected-warning {{overriding method has 'lifetimebound' attribute on parameter 'x', but overridden method is not marked 'lifetimebound'}}
};

struct MultiParamBase {
  virtual const int &f(const int &a, const int &b); // expected-note {{overridden method parameter is here}}
};

struct MultiParamDerived : MultiParamBase {
  const int &f(const int &a, const int &b [[clang::lifetimebound]]) override; // expected-warning {{overriding method has 'lifetimebound' attribute on parameter 'b', but overridden method is not marked 'lifetimebound'}}
};

// --- Pure virtual base ---

struct PureBase {
  virtual const int &g(const int &x) = 0; // expected-note {{overridden method parameter is here}}
};

struct PureDerived : PureBase {
  const int &g(const int &x [[clang::lifetimebound]]) override; // expected-warning {{overriding method has 'lifetimebound' attribute on parameter 'x', but overridden method is not marked 'lifetimebound'}}
};

// --- Template: warn when explicit lifetimebound is added on override ---

template <typename T>
struct TemplateBase {
  virtual const T &get(const T &x); // expected-note {{overridden method parameter is here}}
};

template <typename T>
struct TemplateDerived : TemplateBase<T> {
  const T &get(const T &x [[clang::lifetimebound]]) override; // expected-warning {{overriding method has 'lifetimebound' attribute on parameter 'x', but overridden method is not marked 'lifetimebound'}}
};

void instantiate() {
  TemplateDerived<int> d; // expected-note {{in instantiation of template class 'TemplateDerived<int>' requested here}}
}
