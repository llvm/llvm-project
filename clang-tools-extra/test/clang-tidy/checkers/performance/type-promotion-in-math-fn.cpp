// RUN: %check_clang_tidy %s performance-type-promotion-in-math-fn %t

// CHECK-FIXES: #include <cmath>

double acos(double);
double acosh(double);
double asin(double);
double asinh(double);
double atan2(double, double);
double atan(double);
double atanh(double);
double cbrt(double);
double ceil(double);
double copysign(double, double);
double cos(double);
double cosh(double);
double erfc(double);
double erf(double);
double exp2(double);
double exp(double);
double expm1(double);
double fabs(double);
double fdim(double, double);
double floor(double);
double fma(double, double, double);
double fmax(double, double);
double fmin(double, double);
double fmod(double, double);
double frexp(double, int *);
double hypot(double, double);
double ilogb(double);
double ldexp(double, double);
double lgamma(double);
long long llrint(double);
double log10(double);
double log1p(double);
double log2(double);
double logb(double);
double log(double);
long lrint(double);
double modf(double);
double nearbyint(double);
double nextafter(double, double);
double nexttoward(double, long double);
double pow(double, double);
double remainder(double, double);
double remquo(double, double, int *);
double rint(double);
double round(double);
double scalbln(double, long);
double scalbn(double, int);
double sin(double);
double sinh(double);
double sqrt(double);
double tan(double);
double tanh(double);
double tgamma(double);
double trunc(double);
long long llround(double);
long lround(double);

void check_all_fns() {
  float a, b, c;
  int i;
  long l;
  int *int_ptr;

  acos(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'acos' promotes float to double [performance-type-promotion-in-math-fn]
  // CHECK-FIXES: {{^}}  std::acos(a);{{$}}
  acosh(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'acosh'
  // CHECK-FIXES: {{^}}  std::acosh(a);{{$}}
  asin(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'asin'
  // CHECK-FIXES: {{^}}  std::asin(a);{{$}}
  asinh(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'asinh'
  // CHECK-FIXES: {{^}}  std::asinh(a);{{$}}
  atan2(a, b);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'atan2'
  // CHECK-FIXES: {{^}}  std::atan2(a, b);{{$}}
  atan(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'atan'
  // CHECK-FIXES: {{^}}  std::atan(a);{{$}}
  atanh(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'atanh'
  // CHECK-FIXES: {{^}}  std::atanh(a);{{$}}
  cbrt(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'cbrt'
  // CHECK-FIXES: {{^}}  std::cbrt(a);{{$}}
  ceil(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'ceil'
  // CHECK-FIXES: {{^}}  std::ceil(a);{{$}}
  copysign(a, b);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'copysign'
  // CHECK-FIXES: {{^}}  std::copysign(a, b);{{$}}
  cos(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'cos'
  // CHECK-FIXES: {{^}}  std::cos(a);{{$}}
  cosh(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'cosh'
  // CHECK-FIXES: {{^}}  std::cosh(a);{{$}}
  erf(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'erf'
  // CHECK-FIXES: {{^}}  std::erf(a);{{$}}
  erfc(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'erfc'
  // CHECK-FIXES: {{^}}  std::erfc(a);{{$}}
  exp2(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'exp2'
  // CHECK-FIXES: {{^}}  std::exp2(a);{{$}}
  exp(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'exp'
  // CHECK-FIXES: {{^}}  std::exp(a);{{$}}
  expm1(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'expm1'
  // CHECK-FIXES: {{^}}  std::expm1(a);{{$}}
  fabs(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'fabs'
  // CHECK-FIXES: {{^}}  std::fabs(a);{{$}}
  fdim(a, b);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'fdim'
  // CHECK-FIXES: {{^}}  std::fdim(a, b);{{$}}
  floor(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'floor'
  // CHECK-FIXES: {{^}}  std::floor(a);{{$}}
  fma(a, b, c);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'fma'
  // CHECK-FIXES: {{^}}  std::fma(a, b, c);{{$}}
  fmax(a, b);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'fmax'
  // CHECK-FIXES: {{^}}  std::fmax(a, b);{{$}}
  fmin(a, b);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'fmin'
  // CHECK-FIXES: {{^}}  std::fmin(a, b);{{$}}
  fmod(a, b);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'fmod'
  // CHECK-FIXES: {{^}}  std::fmod(a, b);{{$}}
  frexp(a, int_ptr);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'frexp'
  // CHECK-FIXES: {{^}}  std::frexp(a, int_ptr);{{$}}
  hypot(a, b);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'hypot'
  // CHECK-FIXES: {{^}}  std::hypot(a, b);{{$}}
  ilogb(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'ilogb'
  // CHECK-FIXES: {{^}}  std::ilogb(a);{{$}}
  ldexp(a, b);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'ldexp'
  // CHECK-FIXES: {{^}}  std::ldexp(a, b);{{$}}
  lgamma(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'lgamma'
  // CHECK-FIXES: {{^}}  std::lgamma(a);{{$}}
  llrint(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'llrint'
  // CHECK-FIXES: {{^}}  std::llrint(a);{{$}}
  llround(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'llround'
  // CHECK-FIXES: {{^}}  std::llround(a);{{$}}
  log10(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'log10'
  // CHECK-FIXES: {{^}}  std::log10(a);{{$}}
  log1p(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'log1p'
  // CHECK-FIXES: {{^}}  std::log1p(a);{{$}}
  log2(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'log2'
  // CHECK-FIXES: {{^}}  std::log2(a);{{$}}
  log(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'log'
  // CHECK-FIXES: {{^}}  std::log(a);{{$}}
  logb(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'logb'
  // CHECK-FIXES: {{^}}  std::logb(a);{{$}}
  lrint(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'lrint'
  // CHECK-FIXES: {{^}}  std::lrint(a);{{$}}
  lround(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'lround'
  // CHECK-FIXES: {{^}}  std::lround(a);{{$}}
  nearbyint(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'nearbyint'
  // CHECK-FIXES: {{^}}  std::nearbyint(a);{{$}}
  nextafter(a, b);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'nextafter'
  // CHECK-FIXES: {{^}}  std::nextafter(a, b);{{$}}
  nexttoward(a, b);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'nexttoward'
  // CHECK-FIXES: {{^}}  std::nexttoward(a, b);{{$}}
  pow(a, b);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'pow'
  // CHECK-FIXES: {{^}}  std::pow(a, b);{{$}}
  remainder(a, b);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'remainder'
  // CHECK-FIXES: {{^}}  std::remainder(a, b);{{$}}
  remquo(a, b, int_ptr);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'remquo'
  // CHECK-FIXES: {{^}}  std::remquo(a, b, int_ptr);{{$}}
  rint(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'rint'
  // CHECK-FIXES: {{^}}  std::rint(a);{{$}}
  round(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'round'
  // CHECK-FIXES: {{^}}  std::round(a);{{$}}
  scalbln(a, l);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'scalbln'
  // CHECK-FIXES: {{^}}  std::scalbln(a, l);{{$}}
  scalbn(a, i);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'scalbn'
  // CHECK-FIXES: {{^}}  std::scalbn(a, i);{{$}}
  sin(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'sin'
  // CHECK-FIXES: {{^}}  std::sin(a);{{$}}
  sinh(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'sinh'
  // CHECK-FIXES: {{^}}  std::sinh(a);{{$}}
  sqrt(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'sqrt'
  // CHECK-FIXES: {{^}}  std::sqrt(a);{{$}}
  tan(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'tan'
  // CHECK-FIXES: {{^}}  std::tan(a);{{$}}
  tanh(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'tanh'
  // CHECK-FIXES: {{^}}  std::tanh(a);{{$}}
  tgamma(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'tgamma'
  // CHECK-FIXES: {{^}}  std::tgamma(a);{{$}}
  trunc(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'trunc'
  // CHECK-FIXES: {{^}}  std::trunc(a);{{$}}
}

// nexttoward/nexttowardf are weird -- the second param is always long double.
// So we warn if the first arg is a float, regardless of what the second arg is.
void check_nexttoward() {
  nexttoward(0.f, 0);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'nexttoward'
  // CHECK-FIXES: {{^}}  std::nexttoward(0.f, 0);{{$}}
  nexttoward(0.f, 0l);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'nexttoward'
  // CHECK-FIXES: {{^}}  std::nexttoward(0.f, 0l);{{$}}
  nexttoward(0.f, 0.f);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'nexttoward'
  // CHECK-FIXES: {{^}}  std::nexttoward(0.f, 0.f);{{$}}
  nexttoward(0.f, 0.);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'nexttoward'
  // CHECK-FIXES: {{^}}  std::nexttoward(0.f, 0.);{{$}}

  // No warnings for these.
  nexttoward(0., 0);
  nexttoward(0., 0.f);
  nexttoward(0., 0.);
}

// The second parameter to scalbn and scalbnf is an int, so we don't care what
// type you pass as that argument; we warn iff the first argument is a float.
void check_scalbn() {
  scalbn(0.f, 0);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'scalbn'
  // CHECK-FIXES: {{^}}  std::scalbn(0.f, 0);{{$}}
  scalbn(0.f, static_cast<char>(0));
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'scalbn'
  // CHECK-FIXES: {{^}}  std::scalbn(0.f, static_cast<char>(0));{{$}}

  // No warnings for these.
  scalbn(0., 0);
  scalbn(0., static_cast<char>(0));
}

// scalbln/scalblnf are like scalbn/scalbnf except their second arg is a long.
// Again, doesn't matter what we pass for the second arg; we warn iff the first
// arg is a float.
void check_scalbln() {
  scalbln(0.f, 0);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'scalbln'
  // CHECK-FIXES: {{^}}  std::scalbln(0.f, 0);{{$}}
  scalbln(0.f, 0l);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'scalbln'
  // CHECK-FIXES: {{^}}  std::scalbln(0.f, 0l);{{$}}

  // No warnings for these.
  scalbln(0., 0);
  scalbln(0., 0l);
}

float cosf(float);
double foo(double);         // not a math.h function
float cos(float);           // not a math.h function (wrong signature)
double cos(double, double); // not a math.h function (wrong signature)

namespace std {
void cos(float);
} // namespace std

void check_no_warnings() {
  foo(0.); // no warning because not a math.h function.

  sin(0);        // no warning because arg is an int
  cos(0.);       // no warning because arg is a double
  std::cos(0.f); // no warning because not ::cos.
  cosf(0.f);     // no warning; we expect this to take a float
  cos(0.f);      // does not match the expected signature of ::cos
  cos(0.f, 0.f); // does not match the expected signature of ::cos

  // No warnings because all args are not floats.
  remainder(0., 0.f);
  remainder(0.f, 0.);
  remainder(0, 0.f);
  remainder(0.f, 0);
  fma(0.f, 0.f, 0);
  fma(0.f, 0.f, 0.);
  fma(0.f, 0., 0.f);
  fma(0., 0.f, 0.f);
}
