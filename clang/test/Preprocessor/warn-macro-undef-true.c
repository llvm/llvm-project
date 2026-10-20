// RUN: %clang_cc1 %s -Eonly -std=c89 -verify=undef-true
// RUN: %clang_cc1 %s -Eonly -std=c99 -verify=undef-true
// RUN: %clang_cc1 %s -Eonly -std=c11 -verify=undef-true
// RUN: %clang_cc1 %s -Eonly -std=c17 -verify=undef-true
// RUN: %clang_cc1 %s -Eonly -std=c23 -verify=undef-true

#if __STDC_VERSION__ >= 202311L
/* undef-true-no-diagnostics */
#endif

#define FOO true
#if FOO /* #1 */
#endif
#if __STDC_VERSION__ < 202311L
/* undef-true-warning@#1 {{'true' is not defined, evaluates to 0}} */
#endif

#if true /* #2 */
#endif
#if __STDC_VERSION__ < 202311L
/* undef-true-warning@#2 {{'true' is not defined, evaluates to 0}} */
#endif

#if false || true /* #3 */
#endif
#if __STDC_VERSION__ < 202311L
/* undef-true-warning@#3 {{'true' is not defined, evaluates to 0}} */
#endif

#define true 1

#define FOO true
#if FOO
#endif

#if true
#endif

#if false || true
#endif

#undef true

#define FOO true
#if FOO /* #4 */
#endif
#if __STDC_VERSION__ < 202311L
/* undef-true-warning@#4 {{'true' is not defined, evaluates to 0}} */
#endif

#if true /* #5 */
#endif
#if __STDC_VERSION__ < 202311L
/* undef-true-warning@#5 {{'true' is not defined, evaluates to 0}} */
#endif

#if false || true /* #6 */
#endif
#if __STDC_VERSION__ < 202311L
/* undef-true-warning@#6 {{'true' is not defined, evaluates to 0}} */
#endif

#define true true
#if true /* #7 */
#endif
#if __STDC_VERSION__ < 202311L
/* undef-true-warning@#7 {{'true' is not defined, evaluates to 0}} */
#endif
#undef true

/* Test that #pragma-enabled 'Wundef' can override 'Wundef-true' */
#pragma clang diagnostic warning "-Wundef"
#if true /* #8 */
#endif
#pragma clang diagnostic ignored "-Wundef"
#if __STDC_VERSION__ < 202311L
/* undef-true-warning@#8 {{'true' is not defined, evaluates to 0}} */
#endif
