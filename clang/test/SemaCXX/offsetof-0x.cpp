// RUN: %clang_cc1 -triple x86_64-apple-darwin10.0.0 -fsyntax-only -std=c++11 -verify %s -Winvalid-offsetof

struct NonPOD {
  virtual void f();
  int m;
};

struct P {
  NonPOD fieldThatPointsToANonPODType;
};

void f() {
  int i = __builtin_offsetof(P, fieldThatPointsToANonPODType.m); // expected-warning{{offset of on non-standard-layout type 'P'}}
}

struct StandardLayout {
  int x;
  StandardLayout() {}
};
int o = __builtin_offsetof(StandardLayout, x); // no-warning
