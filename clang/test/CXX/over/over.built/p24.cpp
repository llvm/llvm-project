// RUN: %clang_cc1 -fsyntax-only -std=c++11 -verify %s
// expected-no-diagnostics

struct Variant {
  template <typename T> operator T();
};

Variant getValue();

void testVariant() {
  bool ret1 = getValue() || getValue(); 
  bool ret2 = getValue() && getValue(); 
  bool ret3 = !getValue();
}

struct ExplicitVariant {
  template <typename T> explicit operator T();
};

ExplicitVariant getExplicitValue();

void testExplicitVariant() {
  bool ret1 = getExplicitValue() || getExplicitValue(); 
  bool ret2 = getExplicitValue() && getExplicitValue(); 
  bool ret3 = !getExplicitValue();
}
