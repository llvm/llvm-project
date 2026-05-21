// RUN: %clang_cc1 -verify -Wfenv-access %s
// RUN: %clang_cc1 -verify -Wfenv-access -ffp-exception-behavior=maytrap -DNO_WARN %s
// RUN: %clang_cc1 -verify -Wfenv-access -ffp-exception-behavior=strict -DNO_WARN %s
// RUN: %clang_cc1 -verify -Wfenv-access -triple armv7-linux-gnueabihf -DNO_WARN -DUNSUPPORTED %s

typedef struct {} fenv_t;
typedef unsigned short int fexcept_t;

int feclearexcept(int excepts);
int fegetexceptflag(fexcept_t *flagp, int excepts);
int feraiseexcept(int excepts);
int fesetexceptflag(const fexcept_t *flagp, int excepts);
int fetestexcept(int excepts);
int fegetround(void);
int fesetround(int rounding_mode);
int fegetenv(fenv_t *envp);
int feholdexcept(fenv_t *envp);
int fesetenv(const fenv_t *envp);
int feupdateenv(const fenv_t *envp);

#define FE_INVALID 1

fexcept_t *flagp = 0;
fenv_t *envp = 0;

void test_fenv_access_off(void) {
#ifdef NO_WARN
  // expected-no-diagnostics
  feclearexcept(FE_INVALID);
  fegetexceptflag(flagp, FE_INVALID);
  feraiseexcept(FE_INVALID);
  fesetexceptflag(flagp, FE_INVALID);
  fetestexcept(FE_INVALID);
  fegetround();
  fesetround(0);
  fegetenv(envp);
  feholdexcept(envp);
  fesetenv(envp);
  feupdateenv(envp);
#else 
  feclearexcept(FE_INVALID); // expected-warning {{'feclearexcept' used without enabling floating-point exception behavior; use 'pragma STDC FENV_ACCESS ON' or compile with '-ffp-exception-behavior=maytrap'}}
  fegetexceptflag(flagp, FE_INVALID); // expected-warning {{'fegetexceptflag' used without enabling floating-point exception behavior; use 'pragma STDC FENV_ACCESS ON' or compile with '-ffp-exception-behavior=maytrap'}}
  feraiseexcept(FE_INVALID); // expected-warning {{'feraiseexcept' used without enabling floating-point exception behavior; use 'pragma STDC FENV_ACCESS ON' or compile with '-ffp-exception-behavior=maytrap'}}
  fesetexceptflag(flagp, FE_INVALID); // expected-warning {{'fesetexceptflag' used without enabling floating-point exception behavior; use 'pragma STDC FENV_ACCESS ON' or compile with '-ffp-exception-behavior=maytrap'}}
  fetestexcept(FE_INVALID); // expected-warning {{'fetestexcept' used without enabling floating-point exception behavior; use 'pragma STDC FENV_ACCESS ON' or compile with '-ffp-exception-behavior=maytrap'}}
  fegetround(); // expected-warning {{'fegetround' used without enabling floating-point exception behavior; use 'pragma STDC FENV_ACCESS ON' or compile with '-ffp-exception-behavior=maytrap'}}
  fesetround(0); // expected-warning {{'fesetround' used without enabling floating-point exception behavior; use 'pragma STDC FENV_ACCESS ON' or compile with '-ffp-exception-behavior=maytrap'}}
  fegetenv(envp); // expected-warning {{'fegetenv' used without enabling floating-point exception behavior; use 'pragma STDC FENV_ACCESS ON' or compile with '-ffp-exception-behavior=maytrap'}}
  feholdexcept(envp); // expected-warning {{'feholdexcept' used without enabling floating-point exception behavior; use 'pragma STDC FENV_ACCESS ON' or compile with '-ffp-exception-behavior=maytrap'}}
  fesetenv(envp); // expected-warning {{'fesetenv' used without enabling floating-point exception behavior; use 'pragma STDC FENV_ACCESS ON' or compile with '-ffp-exception-behavior=maytrap'}}
  feupdateenv(envp); // expected-warning {{'feupdateenv' used without enabling floating-point exception behavior; use 'pragma STDC FENV_ACCESS ON' or compile with '-ffp-exception-behavior=maytrap'}}
#endif
}

void test_fenv_access_on(void) {
#ifndef UNSUPPORTED
  #pragma STDC FENV_ACCESS ON
#endif
  fesetround(0);
  feclearexcept(FE_INVALID);
  fegetexceptflag(flagp, FE_INVALID);
  feraiseexcept(FE_INVALID);
  fesetexceptflag(flagp, FE_INVALID);
  fetestexcept(FE_INVALID);
  fegetround();
  fesetround(0);
  fegetenv(envp);
  feholdexcept(envp);
  fesetenv(envp);
  feupdateenv(envp);
}
