// RUN: %clang_cc1 -fsyntax-only -verify %s

template<typename T> void take(T);

void func(float Real, _Complex float Complex) {
  Real += Complex; // expected-error {{assigning to 'float' from incompatible type '_Complex float'}}
  Real += (float)Complex;

  Real = Complex; // expected-error {{implicit conversion from '_Complex float' to 'float' is not permitted in C++}}
  Real = (float)Complex;

  take<float>(Complex); // expected-error {{implicit conversion from '_Complex float' to 'float' is not permitted in C++}}
  take<double>(1.0i); // expected-error {{implicit conversion from '_Complex double' to 'double' is not permitted in C++}}
  take<_Complex float>(Complex);

  // Conversion to bool doesn't actually discard the imaginary part.
  take<bool>(Complex);

  using B = _Complex double;
  B c;
  c *= double();
}
