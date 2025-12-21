// RUN: %clang_cc1 -triple x86_64-apple-darwin -verify -fsyntax-only %s -Wdouble-promotion
// RUN: %clang_cc1 -triple x86_64-apple-darwin -verify -fsyntax-only -x c++ %s -Wdouble-promotion

float ReturnFloatFromDouble(double d) {
  return d;
}

float ReturnFloatFromLongDouble(long double ld) {
  return ld;
}

double ReturnDoubleFromLongDouble(long double ld) {
  return ld;
}

double ReturnDoubleFromFloat(float f) {
  return f;  //expected-warning{{implicit conversion increases floating-point precision: 'float' to 'double'}}
}

long double ReturnLongDoubleFromFloat(float f) {
  return f;  //expected-warning{{implicit conversion increases floating-point precision: 'float' to 'long double'}}
}

long double ReturnLongDoubleFromDouble(double d) {
  return d;  //expected-warning{{implicit conversion increases floating-point precision: 'double' to 'long double'}}
}

double ReturnDoubleFromFloatWithExplicitCast(float f) {
  return (double)f;
}

long double ReturnLongDoubleFromFloatWithExplicitCast(float f) {
  return (long double)f;
}

long double ReturnLongDoubleFromDoubleWithExplicitCast(double d) {
  return (long double)d;
}

void Assignment(float f, double d, long double ld) {
  d = f;  //expected-warning{{implicit conversion increases floating-point precision: 'float' to 'double'}}
  ld = f; //expected-warning{{implicit conversion increases floating-point precision: 'float' to 'long double'}}
  ld = d; //expected-warning{{implicit conversion increases floating-point precision: 'double' to 'long double'}}
  d = (double)f;
  ld = (long double)f;
  ld = (long double)d;
  f = d;
  f = ld;
  d = ld;
}

void AssignmentWithExtraParens(float f, double d, long double ld) {
  d = (f);  //expected-warning{{implicit conversion increases floating-point precision: 'float' to 'double'}}
  ld = (f); //expected-warning{{implicit conversion increases floating-point precision: 'float' to 'long double'}}
  ld = (d); //expected-warning{{implicit conversion increases floating-point precision: 'double' to 'long double'}}
  d = (double)(f);
  ld = (long double)(f);
  ld = (long double)(d);
}

extern void DoubleParameter(double);
extern void LongDoubleParameter(long double);

void ArgumentPassing(float f, double d) {
  DoubleParameter(f); // expected-warning{{implicit conversion increases floating-point precision: 'float' to 'double'}}
  LongDoubleParameter(f); // expected-warning{{implicit conversion increases floating-point precision: 'float' to 'long double'}}
  LongDoubleParameter(d); // expected-warning{{implicit conversion increases floating-point precision: 'double' to 'long double'}}
  DoubleParameter((double)f);
  LongDoubleParameter((long double)f);
  LongDoubleParameter((long double)d);
}

void BinaryOperator(float f, double d, long double ld) {
  f = f * d; // expected-warning{{implicit conversion increases floating-point precision: 'float' to 'double'}}
  f = d * f; // expected-warning{{implicit conversion increases floating-point precision: 'float' to 'double'}}
  f = f * ld; // expected-warning{{implicit conversion increases floating-point precision: 'float' to 'long double'}}
  f = ld * f; // expected-warning{{implicit conversion increases floating-point precision: 'float' to 'long double'}}
  d = d * ld; // expected-warning{{implicit conversion increases floating-point precision: 'double' to 'long double'}}
  d = ld * d; // expected-warning{{implicit conversion increases floating-point precision: 'double' to 'long double'}}
  f = (double)f * d;
  f = d * (double)f;
  f = (long double)f * ld;
  f = ld * (long double)f;
  d = (long double)d * ld;
  d = ld * (long double)d;
}

void MultiplicationAssignment(float f, double d, long double ld) {
  d *= f; // expected-warning{{implicit conversion increases floating-point precision: 'float' to 'double'}}
  ld *= f; // expected-warning{{implicit conversion increases floating-point precision: 'float' to 'long double'}}
  ld *= d; // expected-warning{{implicit conversion increases floating-point precision: 'double' to 'long double'}}
  d *= (double)f;
  ld *= (long double)f;
  ld *= (long double)d;

  // FIXME: These cases should produce warnings as above.
  f *= d;
  f *= ld;
  d *= ld;
}

// FIXME: As with a binary operator, the operands to the conditional operator are
// converted to a common type and should produce a warning.
void ConditionalOperator(float f, double d, long double ld, int i) {
  f = i ? f : d;
  f = i ? d : f;
  f = i ? f : ld;
  f = i ? ld : f;
  d = i ? d : ld;
  d = i ? ld : d;
}
