// RUN: %clang_cc1 -fsyntax-only -Wlifetime-safety-annotation-placement -Wno-dangling -verify %s

#include "Inputs/lifetime-analysis.h"

struct [[gsl::Owner]] Owner {};

struct [[gsl::Pointer()]] View {
  View();
  View(const Owner &o [[clang::lifetimebound]]);
};

struct Plain {};

enum Enum { Enumerator };

struct Mixed {
  Mixed(const int &i [[clang::lifetimebound]]);
};

using IntAlias = int;

Owner *owner_value(Owner o [[clang::lifetimebound]]) { // expected-warning {{'lifetimebound' attribute has no effect on parameter of type 'Owner'}}
  return {};
}

Owner *const_owner_value(const Owner o [[clang::lifetimebound]]) { // expected-warning {{'lifetimebound' attribute has no effect on parameter of type 'const Owner'}}
  return {};
}

Owner *owner_ref(Owner &o [[clang::lifetimebound]]) {
  return &o;
}

const Owner *const_owner_ref(const Owner &o [[clang::lifetimebound]]) {
  return &o;
}

Owner *owner_ptr(Owner *o [[clang::lifetimebound]]) {
  return o;
}

int *scalar_ptr(int *p [[clang::lifetimebound]]) {
  return p;
}

int *scalar_value(int i [[clang::lifetimebound]]) { // expected-warning {{'lifetimebound' attribute has no effect on parameter of type 'int'}}
  return {};
}

int *scalar_alias_value(IntAlias i [[clang::lifetimebound]]) { // expected-warning {{'lifetimebound' attribute has no effect on parameter of type 'IntAlias' (aka 'int')}}
  return {};
}

int *enum_value(Enum e [[clang::lifetimebound]]) { // expected-warning {{'lifetimebound' attribute has no effect on parameter of type 'Enum'}}
  return {};
}

View view_value(View v [[clang::lifetimebound]]) {
  return v;
}

Plain *plain_value(Plain p [[clang::lifetimebound]]) { // expected-warning {{'lifetimebound' attribute has no effect on parameter of type 'Plain'}}
  return {};
}

Mixed *mixed_value(Mixed m [[clang::lifetimebound]]) { // expected-warning {{'lifetimebound' attribute has no effect on parameter of type 'Mixed'}}
  return {};
}

template <class T>
Owner *template_value(T t [[clang::lifetimebound]]) {
  return {};
}

void instantiate_template() {
  Owner o;
  (void)template_value(o);
}

struct S {
  Owner* foo() [[clang::lifetimebound]] { return {}; }
};

std::vector<int *> lifetime_annotated_return(
    const int &i [[clang::lifetimebound]]);

int *context_sensitive_origin_type(
    std::vector<int *> v [[clang::lifetimebound]]) { // expected-warning {{'lifetimebound' attribute has no effect on parameter of type 'std::vector<int *>'}}
  int i = 0;
  lifetime_annotated_return(i);
  return v[0];
}
