// RUN: %clang_cc1 -DWIN -verify -std=c++23 -fsyntax-only  %s
// RUN: %clang_cc1 -verify -std=c++23 -fsyntax-only  %s

// expected-no-diagnostics


#ifdef WIN
#define INFINITY ((float)(1e+300 * 1e+300))
#define NAN      (-(float)(INFINITY * 0.0F))
#else
#define NAN (__builtin_nanf(""))
#define INFINITY (__builtin_inff())
#endif

extern "C" void abort() noexcept;
extern "C" int write(int, const void*, unsigned long);

#define assert(condition) \
  do {			    \
    if (!(condition)) {					\
      write(2, "Assertion failed: ", 18);		\
      write(2, #condition, sizeof(#condition) - 1);	\
      write(2, "\n", 1);				\
      abort();						\
    }							\
  } while (false)

int main() {
    int i;

    // fmin
    static_assert(__builtin_fmin(15.24, 1.3) == 1.3, "");
    static_assert(__builtin_fmin(-0.0, +0.0) == -0.0, "");
    static_assert(__builtin_fmin(+0.0, -0.0) == -0.0, "");
    assert(__builtin_isnan(__builtin_fminf(NAN,NAN)));
    assert(__builtin_isnan(__builtin_fminf(NAN, -1)));
    assert(__builtin_isnan(__builtin_fminf(-INFINITY, 0)));
    assert(__builtin_iszero(__builtin_fminf(+INFINITY, 0)));

    // frexp
    static_assert(__builtin_frexp(123.45, &i) == 0.96445312500000002);
    static_assert(!__builtin_isnan(__builtin_frexp(123.45, &i)), "");
    assert(i==0);
    static_assert(__builtin_iszero(__builtin_frexp(0.0, &i)), "");
    assert(i==0);
    static_assert(__builtin_iszero(__builtin_frexp(-0.0, &i)), "");
    assert(i==0);
    assert(__builtin_isnan(__builtin_frexp(NAN, &i)));
    assert(i==0);
    assert(__builtin_isnan(__builtin_frexp(-NAN, &i)));
    assert(i==0);
    assert(!__builtin_isfinite(__builtin_frexp(INFINITY, &i)));
    assert(i==0);
    assert(!__builtin_isfinite(__builtin_frexp(-INFINITY, &i)));
    assert(i==0);
    return 0;
}
