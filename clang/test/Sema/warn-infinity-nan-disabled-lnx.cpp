// RUN: %clang_cc1 -x c++ -verify=no-inf-no-nan \
// RUN: -triple powerpc64le-unknown-unknown %s -menable-no-infs \
// RUN: -menable-no-nans -std=c++23

// RUN: %clang_cc1 -x c++ -verify=no-inf-no-nan \
// RUN: -triple powerpc64le-unknown-unknown %s -menable-no-infs \
// RUN: -menable-no-nans -funsafe-math-optimizations -std=c++23

// RUN: %clang_cc1 -x c++ -verify=no-fast -triple powerpc64le-unknown-unknown \
// RUN: %s -std=c++23

// RUN: %clang_cc1 -x c++ -verify=no-inf -triple powerpc64le-unknown-unknown %s \
// RUN: -menable-no-infs -std=c++23

// RUN: %clang_cc1 -x c++ -verify=no-inf -triple powerpc64le-unknown-unknown %s \
// RUN: -menable-no-infs -funsafe-math-optimizations -std=c++23

// RUN: %clang_cc1 -x c++ -verify=no-nan -triple powerpc64le-unknown-unknown %s \
// RUN: -menable-no-nans -std=c++23

// RUN: %clang_cc1 -x c++ -verify=no-nan -triple powerpc64le-unknown-unknown %s \
// RUN: -funsafe-math-optimizations -menable-no-nans -std=c++23

// RUN: %clang_cc1 -x c++ -verify=no-fast -triple powerpc64le-unknown-unknown \
// RUN: %s -Wno-nan-infinity-disabled -menable-no-infs -std=c++23

// RUN: %clang_cc1 -x c++ -verify=no-fast -triple powerpc64le-unknown-unknown \
// RUN: %s -Wno-nan-infinity-disabled -menable-no-nans -std=c++23

// no-fast-no-diagnostics

int isunorderedf (float x, float y);
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

template <class _Ty>
class numeric_limits {
public:
    [[nodiscard]] static constexpr _Ty infinity() noexcept {
        return _Ty();
    }
};

template <>
class numeric_limits<float>  {
public:
    [[nodiscard]] static constexpr float infinity() noexcept {
        return __builtin_huge_val();
    }
};
template <>
class numeric_limits<double>  {
public:
    [[nodiscard]] static constexpr double infinity() noexcept {
        return __builtin_huge_val();
    }
};

int compareit(float a, float b) {
  volatile int i, j, k, l, m, n, o, p;
// no-inf-no-nan-warning@+2 {{use of infinity via a macro is undefined behavior due to the currently enabled floating-point options}}
// no-inf-warning@+1 {{use of infinity via a macro is undefined behavior due to the currently enabled floating-point options}}
  i = a == INFINITY;

// no-inf-no-nan-warning@+2 {{use of infinity via a macro is undefined behavior due to the currently enabled floating-point options}}
// no-inf-warning@+1 {{use of infinity via a macro is undefined behavior due to the currently enabled floating-point options}}
  j = INFINITY == a;

// no-inf-no-nan-warning@+4 {{use of NaN is undefined behavior due to the currently enabled floating-point options}}
// no-inf-no-nan-warning@+3 {{use of NaN via a macro is undefined behavior due to the currently enabled floating-point options}}
// no-nan-warning@+2 {{use of NaN via a macro is undefined behavior due to the currently enabled floating-point options}}
// no-nan-warning@+1 {{use of NaN is undefined behavior due to the currently enabled floating-point options}}
  i = a == NAN;

// no-inf-no-nan-warning@+4 {{use of NaN via a macro is undefined behavior due to the currently enabled floating-point options}}
// no-inf-no-nan-warning@+3 {{use of NaN is undefined behavior due to the currently enabled floating-point options}}
// no-nan-warning@+2 {{use of NaN is undefined behavior due to the currently enabled floating-point options}}
// no-nan-warning@+1 {{use of NaN via a macro is undefined behavior due to the currently enabled floating-point options}}
  j = NAN == a;

// no-inf-no-nan-warning@+2 {{use of infinity via a macro is undefined behavior due to the currently enabled floating-point options}}
// no-inf-warning@+1 {{use of infinity via a macro is undefined behavior due to the currently enabled floating-point options}}
  j = INFINITY <= a;

// no-inf-no-nan-warning@+2 {{use of infinity via a macro is undefined behavior due to the currently enabled floating-point options}}
// no-inf-warning@+1 {{use of infinity via a macro is undefined behavior due to the currently enabled floating-point options}}
  j = INFINITY < a;

// no-inf-no-nan-warning@+4 {{use of NaN via a macro is undefined behavior due to the currently enabled floating-point options}}
// no-inf-no-nan-warning@+3 {{use of NaN is undefined behavior due to the currently enabled floating-point options}}
// no-nan-warning@+2 {{use of NaN via a macro is undefined behavior due to the currently enabled floating-point options}}
// no-nan-warning@+1 {{use of NaN is undefined behavior due to the currently enabled floating-point options}}
  j = a > NAN;

// no-inf-no-nan-warning@+4 {{use of NaN via a macro is undefined behavior due to the currently enabled floating-point options}}
// no-inf-no-nan-warning@+3 {{use of NaN is undefined behavior due to the currently enabled floating-point options}}
// no-nan-warning@+2 {{use of NaN is undefined behavior due to the currently enabled floating-point options}}
// no-nan-warning@+1 {{use of NaN via a macro is undefined behavior due to the currently enabled floating-point options}}
  j = a >= NAN;

// no-inf-no-nan-warning@+2 {{use of infinity is undefined behavior due to the currently enabled floating-point options}}
// no-inf-warning@+1 {{use of infinity is undefined behavior due to the currently enabled floating-point options}}
  k = std::isinf(a);

// no-inf-no-nan-warning@+2 {{use of NaN is undefined behavior due to the currently enabled floating-point option}}
// no-nan-warning@+1 {{use of NaN is undefined behavior due to the currently enabled floating-point options}}
  l = std::isnan(a);

// no-inf-no-nan-warning@+2 {{use of infinity is undefined behavior due to the currently enabled floating-point options}}
// no-inf-warning@+1 {{use of infinity is undefined behavior due to the currently enabled floating-point options}}
  o = std::isfinite(a);

// no-inf-no-nan-warning@+2 {{use of infinity is undefined behavior due to the currently enabled floating-point options}}
// no-inf-warning@+1 {{use of infinity is undefined behavior due to the currently enabled floating-point options}}
  m = __builtin_isinf(a);

// no-inf-no-nan-warning@+2 {{use of NaN is undefined behavior due to the currently enabled floating-point options}}
// no-nan-warning@+1 {{use of NaN is undefined behavior due to the currently enabled floating-point options}}
  n = __builtin_isnan(a);

// no-inf-no-nan-warning@+2 {{use of infinity is undefined behavior due to the currently enabled floating-point options}}
// no-inf-warning@+1 {{use of infinity is undefined behavior due to the currently enabled floating-point options}}
  p = __builtin_isfinite(a);

// These should NOT warn, since they are not using NaN or infinity.
  j = a > 1.1;
  j = b < 1.1;
  j = a >= 1.1;
  j = b <= 1.1;
  j = isunorderedf(a, b);

#ifndef INFINITY
  j = a;
#endif
#ifndef NAN
  j = b;
#endif
#ifdef INFINITY
  j = a;
#endif
#ifdef NAN
  j = b;
#endif
#if defined(INFINITY)
  j = a;
#elifndef(INFINITY)
  j = b;
#endif
#if defined(INFINITY)
  j = a;
#elifndef(NAN)
  j = b;
#endif
#if defined(NAN)
  j = a;
#elifndef(INFINITY)
  j = b;
#endif

// no-inf-no-nan-warning@+4 {{use of NaN via a macro is undefined behavior due to the currently enabled floating-point options}}
// no-inf-no-nan-warning@+3 {{use of NaN is undefined behavior due to the currently enabled floating-point options}}
// no-nan-warning@+2 {{use of NaN via a macro is undefined behavior due to the currently enabled floating-point options}}
// no-nan-warning@+1 {{use of NaN is undefined behavior due to the currently enabled floating-point options}}
  j = isunorderedf(a, NAN);

// no-inf-no-nan-warning@+2 {{use of infinity via a macro is undefined behavior due to the currently enabled floating-point options}}
// no-inf-warning@+1 {{use of infinity via a macro is undefined behavior due to the currently enabled floating-point options}}
  j = isunorderedf(a, INFINITY);

// no-inf-no-nan-warning@+6 {{use of NaN via a macro is undefined behavior due to the currently enabled floating-point options}}
// no-inf-no-nan-warning@+5 {{use of NaN is undefined behavior due to the currently enabled floating-point options}}
// no-inf-no-nan-warning@+4 {{use of NaN is undefined behavior due to the currently enabled floating-point options}}
// no-nan-warning@+3 {{use of NaN via a macro is undefined behavior due to the currently enabled floating-point options}}
// no-nan-warning@+2 {{use of NaN is undefined behavior due to the currently enabled floating-point options}}
// no-nan-warning@+1 {{use of NaN is undefined behavior due to the currently enabled floating-point options}}
  i = std::isunordered(a, NAN);

// no-inf-no-nan-warning@+4 {{use of infinity via a macro is undefined behavior due to the currently enabled floating-point options}}
// no-inf-no-nan-warning@+3 {{use of NaN is undefined behavior due to the currently enabled floating-point options}}
// no-inf-warning@+2 {{use of infinity via a macro is undefined behavior due to the currently enabled floating-point options}}
// no-nan-warning@+1 {{use of NaN is undefined behavior due to the currently enabled floating-point options}}
  i = std::isunordered(a, INFINITY);

// no-inf-no-nan-warning@+2 {{use of infinity is undefined behavior due to the currently enabled floating-point options}}
// no-inf-warning@+1 {{use of infinity is undefined behavior due to the currently enabled floating-point options}}
  double y = i * numeric_limits<double>::infinity();

// no-inf-no-nan-warning@+2 {{use of infinity is undefined behavior due to the currently enabled floating-point options}}
// no-inf-warning@+1 {{use of infinity is undefined behavior due to the currently enabled floating-point options}}
  j = numeric_limits<float>::infinity();
  return 0;

}
