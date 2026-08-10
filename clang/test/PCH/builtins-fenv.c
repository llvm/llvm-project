// Test this without pch.
// RUN: %clang_cc1 -include %S/builtins-fenv.h -fsyntax-only -verify %s

// Test with pch.
// RUN: %clang_cc1 -emit-pch -o %t %S/builtins-fenv.h
// RUN: %clang_cc1 -include-pch %t -fsyntax-only -verify %s 

// expected-no-diagnostics
fexcept_t *flagp = 0;
fenv_t *envp = 0;

void f(void) {
  #pragma STDC FENV_ACCESS ON
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
