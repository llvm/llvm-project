// RUN: %clang_cc1 %s -Eonly -Wundef -verify=undef
// RUN: %clang_cc1 %s -Eonly -Wundef-prefix=A,BC -verify=undef-prefix
// RUN: %clang_cc1 %s -Eonly -Wundef -Wundef-prefix=A,BC -verify=both
// RUN: %clang_cc1 %s -Eonly -Werror=undef -verify=undef-error
// RUN: %clang_cc1 %s -Eonly -Werror=undef-prefix -Wundef-prefix=A,BC -verify=undef-prefix-error
// RUN: %clang_cc1 %s -Eonly -Werror=undef -Wundef-prefix=A,BC -verify=both-error

extern int x;

#if AB // #1
#endif
// undef-warning@#1 {{'AB' is not defined, evaluates to 0}}
// undef-prefix-warning@#1 {{'AB' is not defined, evaluates to 0}}
// both-warning@#1 {{'AB' is not defined, evaluates to 0}}
// undef-error-error@#1 {{'AB' is not defined, evaluates to 0}}
// undef-prefix-error-error@#1 {{'AB' is not defined, evaluates to 0}}
// both-error-error@#1 {{'AB' is not defined, evaluates to 0}}

#if B // #2
#endif
// undef-warning@#2 {{'B' is not defined, evaluates to 0}}
// no warning for undef-prefix
// both-warning@#2 {{'B' is not defined, evaluates to 0}}
// undef-error-error@#2 {{'B' is not defined, evaluates to 0}}
// no error for undef-prefix
// both-error-error@#2 {{'B' is not defined, evaluates to 0}}

#define BC 0
#if BC // no warning/error
#endif

#undef BC
#if BC // #3
#endif
// undef-warning@#3 {{'BC' is not defined, evaluates to 0}}
// undef-prefix-warning@#3 {{'BC' is not defined, evaluates to 0}}
// both-warning@#3 {{'BC' is not defined, evaluates to 0}}
// undef-error-error@#3 {{'BC' is not defined, evaluates to 0}}
// undef-prefix-error-error@#3 {{'BC' is not defined, evaluates to 0}}
// both-error-error@#3 {{'BC' is not defined, evaluates to 0}}

// Test that #pragma-enabled 'Wundef' can override 'Wundef-prefix'
#pragma clang diagnostic error "-Wundef"

#if C // #4
#endif
// undef-error@#4 {{'C' is not defined, evaluates to 0}}
// undef-prefix-error@#4 {{'C' is not defined, evaluates to 0}}
// both-error@#4 {{'C' is not defined, evaluates to 0}}
// undef-error-error@#4 {{'C' is not defined, evaluates to 0}}
// undef-prefix-error-error@#4 {{'C' is not defined, evaluates to 0}}
// both-error-error@#4 {{'C' is not defined, evaluates to 0}}
