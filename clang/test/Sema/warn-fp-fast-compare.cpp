// RUN: %clang_cc1 -x c++ -verify -triple powerpc64le-unknown-unknown %s \
// RUN: -menable-no-infs -menable-no-nans  -DFAST=1

// RUN: %clang_cc1 -x c++ -verify -triple powerpc64le-unknown-unknown %s \
// RUN: -DNOFAST=1

// RUN: %clang_cc1 -x c++ -verify -triple powerpc64le-unknown-unknown %s \
// RUN: -menable-no-infs -DNO_INFS=1

// RUN: %clang_cc1 -x c++ -verify -triple powerpc64le-unknown-unknown %s \
// RUN: -menable-no-nans -DNO_NANS=1

int isunorderedf (float x, float y);
#if NOFAST
// expected-no-diagnostics
#endif
extern "C++" {
namespace std __attribute__((__visibility__("default"))) {
  bool
  isinf(float __x);
  bool
  isinf(double __x);
  bool
  isinf(long double __x);
  bool
  isnan(float __x);
  bool
  isnan(double __x);
  bool
  isnan(long double __x);
bool
  isfinite(float __x);
  bool
  isfinite(double __x);
  bool
  isfinte(long double __x);
 bool
  isunordered(float __x, float __y);
  bool
  isunordered(double __x, double __y);
  bool
  isunordered(long double __x, long double __y);
} // namespace )
}
#define NAN (__builtin_nanf(""))
#define INFINITY (__builtin_inff())

int compareit(float a, float b) {
  volatile int i, j, k, l, m, n, o, p;
#if FAST
// expected-warning@+5 {{explicit comparison with infinity in fast floating point mode}}
#endif
#if NO_INFS
// expected-warning@+2 {{explicit comparison with infinity in fast floating point mode}}
#endif
  i = a == INFINITY;
#if FAST
// expected-warning@+5 {{explicit comparison with infinity in fast floating point mode}}
#endif
#if NO_INFS
// expected-warning@+2 {{explicit comparison with infinity in fast floating point mode}}
#endif
  j = INFINITY == a;
#if FAST
// expected-warning@+5 {{explicit comparison with NaN in fast floating point mode}}
#endif
#if NO_NANS
// expected-warning@+2 {{explicit comparison with NaN in fast floating point mode}}
#endif
  i = a == NAN;
#if FAST
// expected-warning@+5 {{explicit comparison with NaN in fast floating point mode}}
#endif
#if NO_NANS
// expected-warning@+2 {{explicit comparison with NaN in fast floating point mode}}
#endif
  j = NAN == a;
#if FAST
// expected-warning@+5 {{explicit comparison with infinity in fast floating point mode}}
#endif
#if NO_INFS
// expected-warning@+2 {{explicit comparison with infinity in fast floating point mode}}
#endif
  j = INFINITY <= a;
#if FAST
// expected-warning@+5 {{explicit comparison with infinity in fast floating point mode}}
#endif
#if NO_INFS
// expected-warning@+2 {{explicit comparison with infinity in fast floating point mode}}
#endif
  j = INFINITY < a;
#if FAST
// expected-warning@+5 {{explicit comparison with NaN in fast floating point mode}}
#endif
#if NO_NANS
// expected-warning@+2 {{explicit comparison with NaN in fast floating point mode}}
#endif
  j = a > NAN;
#if FAST
// expected-warning@+5 {{explicit comparison with NaN in fast floating point mode}}
#endif
#if NO_NANS
// expected-warning@+2 {{explicit comparison with NaN in fast floating point mode}}
#endif
  j = a >= NAN;
#if FAST
// expected-warning@+5 {{explicit comparison with infinity in fast floating point mode}}
#endif
#if NO_INFS
// expected-warning@+2 {{explicit comparison with infinity in fast floating point mode}}
#endif
k = std::isinf(a);
#if FAST
// expected-warning@+5 {{explicit comparison with NaN in fast floating point mode}}
#endif
#if NO_NANS
// expected-warning@+2 {{explicit comparison with NaN in fast floating point mode}}
#endif
  l = std::isnan(a);
#if FAST
// expected-warning@+5 {{explicit comparison with infinity in fast floating point mode}}
#endif
#if NO_INFS
//expected-warning@+2 {{explicit comparison with infinity in fast floating point mode}}
#endif
  o = std::isfinite(a);
#if FAST
// expected-warning@+5 {{explicit comparison with infinity in fast floating point mode}}
#endif
#if NO_INFS
// expected-warning@+2 {{explicit comparison with infinity in fast floating point mode}}
#endif
  m = __builtin_isinf(a);
#if FAST
// expected-warning@+5 {{explicit comparison with NaN in fast floating point mode}}
#endif
#if NO_NANS
// expected-warning@+2 {{explicit comparison with NaN in fast floating point mode}}
#endif
  n = __builtin_isnan(a);
#if FAST
//expected-warning@+5 {{explicit comparison with infinity in fast floating point mode}}
#endif
#if NO_INFS
//expected-warning@+2 {{explicit comparison with infinity in fast floating point mode}}
#endif
  p = __builtin_isfinite(a);

  // These should NOT warn, since they are not comparing with NaN or infinity.
  j = a > 1.1;
  j = b < 1.1;
  j = a >= 1.1;
  j = b <= 1.1;
  j = isunorderedf(a, NAN);
  j = isunorderedf(a, INFINITY);
#if FAST
// expected-warning@+5 {{explicit comparison with NaN in fast floating point mode}}
#endif
#if NO_NANS
// expected-warning@+2 {{explicit comparison with NaN in fast floating point mode}}
#endif
  i = std::isunordered(a, NAN);
#if FAST
// expected-warning@+5 {{explicit comparison with NaN in fast floating point mode}}
#endif
#if NO_NANS
// expected-warning@+2 {{explicit comparison with NaN in fast floating point mode}}
#endif
  i = std::isunordered(a, INFINITY);
  return 0;
}  
