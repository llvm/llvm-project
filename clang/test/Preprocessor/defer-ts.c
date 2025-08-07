// RUN: %clang_cc1 -fdefer-ts -fsyntax-only %s
#if __STDC_DEFER_TS25755__ != 1
#  error Should have defined __STDC_DEFER_TS25755__
#endif
