// RUN: %clang_cc1 -verify -Wfenv-access %s
// RUN: %clang_cc1 -verify -Wfenv-access -ffp-exception-behavior=maytrap -DNO_WARN %s
// RUN: %clang_cc1 -verify -Wfenv-access -ffp-exception-behavior=strict -DNO_WARN %s

int feclearexcept(int excepts);
int feraiseexcept(int excepts);
int fetestexcept(int excepts);
int fegetround(void);
int fesetround(int rounding_mode);

#define FE_INVALID 1

void test_fenv_access_off(void) {
#ifdef NO_WARN
  // expected-no-diagnostics
  feclearexcept(FE_INVALID);
  feraiseexcept(FE_INVALID);
  fetestexcept(FE_INVALID);
  fegetround();
  fesetround(0);
#else 
  feclearexcept(FE_INVALID); // expected-warning {{'feclearexcept' used without enabling floating-point exception behavior; use 'pragma STDC FENV_ACCESS ON' or compile with '-ffp-exception-behavior=maytrap'}}
  feraiseexcept(FE_INVALID); // expected-warning {{'feraiseexcept' used without enabling floating-point exception behavior; use 'pragma STDC FENV_ACCESS ON' or compile with '-ffp-exception-behavior=maytrap'}}
  fetestexcept(FE_INVALID); // expected-warning {{'fetestexcept' used without enabling floating-point exception behavior; use 'pragma STDC FENV_ACCESS ON' or compile with '-ffp-exception-behavior=maytrap'}}
  fegetround(); // expected-warning {{'fegetround' used without enabling floating-point exception behavior; use 'pragma STDC FENV_ACCESS ON' or compile with '-ffp-exception-behavior=maytrap'}}
  fesetround(0); // expected-warning {{'fesetround' used without enabling floating-point exception behavior; use 'pragma STDC FENV_ACCESS ON' or compile with '-ffp-exception-behavior=maytrap'}}
#endif
}

void test_fenv_access_on(void) {
  #pragma STDC FENV_ACCESS ON
  feclearexcept(FE_INVALID);
  feraiseexcept(FE_INVALID);
  fetestexcept(FE_INVALID);
  fegetround();
  fesetround(0);
}
