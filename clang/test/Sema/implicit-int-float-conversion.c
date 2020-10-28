// RUN: %clang_cc1 %s -verify -Wno-constant-conversion
// RUN: %clang_cc1 %s -verify -Wno-constant-conversion -Wno-implicit-int-float-conversion -Wimplicit-const-int-float-conversion
// RUN: %clang_cc1 %s -DNONCONST=1 -verify -Wno-constant-conversion -Wimplicit-int-float-conversion

#ifdef NONCONST
long testReturn(long a, float b) {
  return a + b; // expected-warning {{implicit conversion from 'long' to 'float' may lose precision}}
}
#endif

void testAssignment() {
  float f = 222222;
  double b = 222222222222L;

  float ff = 222222222222L;    // expected-warning {{changes value from 222222222222 to 222222221312}}
  float ffff = 222222222222UL; // expected-warning {{changes value from 222222222222 to 222222221312}}

  long l = 222222222222L;
#ifdef NONCONST
  float fff = l; // expected-warning {{implicit conversion from 'long' to 'float' may lose precision}}
#endif
}

void testExpression() {
  float a = 0.0f;

  float b = 222222222222L + a; // expected-warning {{changes value from 222222222222 to 222222221312}}

  float g = 22222222 + 22222222;
  float c = 22222222 + 22222223; // expected-warning {{implicit conversion from 'int' to 'float' changes value from 44444445 to 44444444}}

  int i = 0;
#ifdef NONCONST
  float d = i + a; // expected-warning {{implicit conversion from 'int' to 'float' may lose precision}}
#endif

  double e = 0.0;
  double f = i + e;
}

void testCNarrowing() {
  // Since this is a C file. C++11 narrowing is not in effect.
  // In this case, we should issue warnings.
  float a = {222222222222L}; // expected-warning {{changes value from 222222222222 to 222222221312}}

  long b = 222222222222L;
#ifdef NONCONST
  float c = {b}; // expected-warning {{implicit conversion from 'long' to 'float' may lose precision}}
#endif
}
