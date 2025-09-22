// RUN: %clang_cc1 -triple x86_64-apple-darwin -verify -fsyntax-only %s -Wdouble-promotion

using LongDouble = long double;

double ReturnDoubleFromFloatWithExplicitCast(float f) {
  return static_cast<double>(f);
}

long double ReturnLongDoubleFromFloatWithExplicitCast(float f) {
  return static_cast<long double>(f);
}

long double ReturnLongDoubleFromDoubleWithExplicitCast(double d) {
  return static_cast<long double>(d);
}

double ReturnDoubleFromFloatWithExplicitListInitialization(float f) {
  return double{f};
}

long double ReturnLongDoubleFromFloatWithExplicitListInitialization(float f) {
  return LongDouble{f};
}

long double ReturnLongDoubleFromDoubleWithExplicitListInitialization(double d) {
  return LongDouble{d};
}

double ReturnDoubleFromFloatWithFunctionStyleCast(float f) {
  return double(f);
}

long double ReturnLongDoubleFromFloatWithFunctionStyleCast(float f) {
  return LongDouble(f);
}

long double ReturnLongDoubleFromDoubleWithFunctionStyleCast(double d) {
  return LongDouble(d);
}

void InitializationWithParens(float f, double d) {
  {
    double d(f);  // expected-warning{{implicit conversion increases floating-point precision: 'float' to 'double'}}
    long double ld0(f);  // expected-warning{{implicit conversion increases floating-point precision: 'float' to 'long double'}}
    long double ld1(d);  // expected-warning{{implicit conversion increases floating-point precision: 'double' to 'long double'}}
  }
  {
    double d(static_cast<double>(f));
    long double ld0(static_cast<long double>(f));
    long double ld1(static_cast<long double>(d));
  }
  {
    double d(double{f});
    long double ld0(LongDouble{f});
    long double ld1(LongDouble{d});
  }
  {
    double d((double(f)));
    long double ld0((LongDouble(f)));
    long double ld1((LongDouble(d)));
  }
}

void InitializationWithBraces(float f, double d) {
  {
    double d{f};  // expected-warning{{implicit conversion increases floating-point precision: 'float' to 'double'}}
    long double ld0{f};  // expected-warning{{implicit conversion increases floating-point precision: 'float' to 'long double'}}
    long double ld1{d};  // expected-warning{{implicit conversion increases floating-point precision: 'double' to 'long double'}}
  }
  {
    double d{static_cast<double>(f)};
    long double ld0{static_cast<long double>(f)};
    long double ld1{static_cast<long double>(d)};
  }
  {
    double d{double{f}};
    long double ld0{LongDouble{f}};
    long double ld1{LongDouble{d}};
  }
  {
    double d{double(f)};
    long double ld0{LongDouble(f)};
    long double ld1{LongDouble(d)};
  }
}

void Assignment(float f, double d, long double ld) {
  d = static_cast<double>(f);
  ld = static_cast<long double>(f);
  ld = static_cast<long double>(d);
  d = double{f};
  ld = LongDouble{f};
  ld = LongDouble{d};
  d = double(f);
  ld = LongDouble(f);
  ld = LongDouble(d);
}

extern void DoubleParameter(double);
extern void LongDoubleParameter(long double);

void ArgumentPassing(float f, double d) {
  DoubleParameter(static_cast<double>(f));
  LongDoubleParameter(static_cast<long double>(f));
  LongDoubleParameter(static_cast<long double>(d));
  DoubleParameter(double{f});
  LongDoubleParameter(LongDouble{f});
  LongDoubleParameter(LongDouble{d});
  DoubleParameter(double(f));
  LongDoubleParameter(LongDouble(f));
  LongDoubleParameter(LongDouble(d));
}

void BinaryOperator(float f, double d, long double ld) {
  f = static_cast<double>(f) * d;
  f = d * static_cast<double>(f);
  f = static_cast<long double>(f) * ld;
  f = ld * static_cast<long double>(f);
  d = static_cast<long double>(d) * ld;
  d = ld * static_cast<long double>(d);
  f = double{f} * d;
  f = d * double{f};
  f = LongDouble{f} * ld;
  f = ld * LongDouble{f};
  d = LongDouble{d} * ld;
  d = ld * LongDouble{d};
  f = double(f) * d;
  f = d * double(f);
  f = LongDouble(f) * ld;
  f = ld * LongDouble(f);
  d = LongDouble(d) * ld;
  d = ld * LongDouble(d);
}

void MultiplicationAssignment(float f, double d, long double ld) {
  d *= static_cast<double>(f);
  ld *= static_cast<long double>(f);
  ld *= static_cast<long double>(d);
  d *= double{f};
  ld *= LongDouble{f};
  ld *= LongDouble{d};
  d *= double(f);
  ld *= LongDouble(f);
  ld *= LongDouble(d);
}
