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

void AssignmentWithExtraParens(float f, double d, long double ld) {
  d = static_cast<double>((f));
  ld = static_cast<long double>((f));
  ld = static_cast<long double>((d));
  d = double{(f)};
  ld = LongDouble{(f)};
  ld = LongDouble{(d)};
  d = double((f));
  ld = LongDouble((f));
  ld = LongDouble((d));
}

void AssignmentWithExtraBraces(float f, double d, long double ld) {
  d = double{{f}};  // expected-warning{{too many braces around scalar initializer}}
  ld = LongDouble{{f}};  // expected-warning{{too many braces around scalar initializer}}
  ld = LongDouble{{d}};  // expected-warning{{too many braces around scalar initializer}}
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

struct ConstructWithDouble {
  ConstructWithDouble(double);
};

struct ConstructWithLongDouble {
  ConstructWithLongDouble(long double);
};

void Construct(float f, double d) {
  ConstructWithDouble{f};  // expected-warning{{implicit conversion increases floating-point precision: 'float' to 'double'}}
  ConstructWithLongDouble{f};  // expected-warning{{implicit conversion increases floating-point precision: 'float' to 'long double'}}
  ConstructWithLongDouble{d};  // expected-warning{{implicit conversion increases floating-point precision: 'double' to 'long double'}}
  ConstructWithDouble{static_cast<double>(f)};
  ConstructWithLongDouble{static_cast<long double>(f)};
  ConstructWithLongDouble{static_cast<long double>(d)};
  ConstructWithDouble{double{f}};
  ConstructWithLongDouble{LongDouble{f}};
  ConstructWithLongDouble{LongDouble{d}};
  ConstructWithDouble{double(f)};
  ConstructWithLongDouble{LongDouble(f)};
  ConstructWithLongDouble{LongDouble(d)};
}

template <class T> T ReturnTFromFloat(float f) {
  return f;  // expected-warning{{implicit conversion increases floating-point precision: 'float' to 'double'}} \
             // expected-warning{{implicit conversion increases floating-point precision: 'float' to 'long double'}}
}

template <class T> T ReturnTFromDouble(double d) {
  return d;  // expected-warning{{implicit conversion increases floating-point precision: 'double' to 'long double'}}
}

template <class T> T ReturnTFromFloatWithStaticCast(float f) {
  return static_cast<T>(f);
}

template <class T> T ReturnTFromDoubleWithStaticCast(double d) {
  return static_cast<T>(d);
}

template <class T> T ReturnTFromFloatWithExplicitListInitialization(float f) {
  return T{f};
}

template <class T> T ReturnTFromDoubleWithExplicitListInitialization(double d) {
  return T{d};
}

template <class T> T ReturnTFromFloatWithFunctionStyleCast(float f) {
  return T(f);
}

template <class T> T ReturnTFromDoubleWithFunctionStyleCast(double d) {
  return T(d);
}

void TestTemplate(float f, double d) {
  ReturnTFromFloat<double>(f);  // expected-note{{in instantiation of function template specialization 'ReturnTFromFloat<double>' requested here}}
  ReturnTFromFloat<long double>(f);  // expected-note{{in instantiation of function template specialization 'ReturnTFromFloat<long double>' requested here}}
  ReturnTFromDouble<long double>(d);  // expected-note{{in instantiation of function template specialization 'ReturnTFromDouble<long double>' requested here}}
  ReturnTFromFloatWithStaticCast<double>(f);
  ReturnTFromFloatWithStaticCast<long double>(f);
  ReturnTFromDoubleWithStaticCast<long double>(d);
  ReturnTFromFloatWithExplicitListInitialization<double>(f);
  ReturnTFromFloatWithExplicitListInitialization<long double>(f);
  ReturnTFromDoubleWithExplicitListInitialization<long double>(d);
  ReturnTFromFloatWithFunctionStyleCast<double>(f);
  ReturnTFromFloatWithFunctionStyleCast<long double>(f);
  ReturnTFromDoubleWithFunctionStyleCast<long double>(d);
}

struct MemberInitializerListParens {
  double m_d;
  long double m_ld0;
  long double m_ld1;
  MemberInitializerListParens(float f, double d):
    m_d(f),  // expected-warning{{implicit conversion increases floating-point precision: 'float' to 'double'}}
    m_ld0(f),  // expected-warning{{implicit conversion increases floating-point precision: 'float' to 'long double'}}
    m_ld1(d)  // expected-warning{{implicit conversion increases floating-point precision: 'double' to 'long double'}}
  {}
};

struct MemberInitializerListBraces {
  double m_d;
  long double m_ld0;
  long double m_ld1;
  MemberInitializerListBraces(float f, double d):
    m_d{f},  // expected-warning{{implicit conversion increases floating-point precision: 'float' to 'double'}}
    m_ld0{f},  // expected-warning{{implicit conversion increases floating-point precision: 'float' to 'long double'}}
    m_ld1{d}  // expected-warning{{implicit conversion increases floating-point precision: 'double' to 'long double'}}
  {}
};
