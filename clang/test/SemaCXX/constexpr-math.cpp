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

int main() {
    int i;

    // fmin
    static_assert(__builtin_fmin(15.24, 1.3) == 1.3, "");
    static_assert(__builtin_fmin(-0.0, +0.0) == -0.0, "");
    static_assert(__builtin_fmin(+0.0, -0.0) == -0.0, "");
    static_assert(__builtin_fminf(NAN, -1) == -1, "");
    static_assert(__builtin_fminf(+INFINITY, 0) == 0, "");
    static_assert(__builtin_isinf(__builtin_fminf(-INFINITY, 0)), "");
    static_assert(__builtin_isnan(__builtin_fminf(NAN,NAN)), "");

    // frexp
    static_assert(__builtin_frexp(123.45, &i) == 0.96445312500000002);
    static_assert(!__builtin_isnan(__builtin_frexp(123.45, &i)), "");
    static_assert(__builtin_iszero(__builtin_frexp(0.0, &i)), "");
    static_assert(__builtin_iszero(__builtin_frexp(-0.0, &i)), "");
    static_assert(__builtin_isnan(__builtin_frexp(NAN, &i)));
    static_assert(__builtin_isnan(__builtin_frexp(-NAN, &i)));
    static_assert(!__builtin_isfinite(__builtin_frexp(INFINITY, &i)));
    static_assert(!__builtin_isfinite(__builtin_frexp(-INFINITY, &i)));

    return 0;
}
