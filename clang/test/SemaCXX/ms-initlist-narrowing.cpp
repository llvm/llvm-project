// RUN: %clang_cc1 %s -fsyntax-only -verify -fms-extensions -fms-compatibility-version=19.0 -std=c++11

int ai[] = { 1, 2.0 };  // expected-error {{type 'double' cannot be narrowed to 'int' in initializer list}} expected-note {{silence}}

template<typename T>
struct Agg {
  T t;
};

void f(int i) {
  // Constant expression.
  Agg<float> f8 = {1E50};  // expected-error {{constant expression evaluates to 1.000000e+50 which cannot be narrowed to type 'float'}} expected-note {{silence}}

  // Non-constant expression.
  double d = 1.0;
  Agg<float> f2 = {d};  // expected-error {{non-constant-expression cannot be narrowed from type 'double' to 'float'}} expected-note {{silence}}
}
