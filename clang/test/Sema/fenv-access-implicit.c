// RUN: %clang_cc1 -verify -Wfenv-access %s

typedef struct {} fenv_t;
typedef unsigned short int fexcept_t;

fexcept_t *flagp = 0;
fenv_t *envp = 0;

#define FE_INVALID 1

void test_fenv_access_undeclared(void) {
  #pragma STDC FENV_ACCESS ON
  feclearexcept(FE_INVALID); // expected-note {{include the header <fenv.h> or explicitly provide a declaration for 'feclearexcept'}} \
   expected-error {{call to undeclared library function 'feclearexcept' with type 'int (int)'; ISO C99 and later do not support implicit function declarations}}
  fegetexceptflag(flagp, FE_INVALID); // expected-note {{include the header <fenv.h> or explicitly provide a declaration for 'fegetexceptflag'}} \
   expected-error {{call to undeclared library function 'fegetexceptflag' with type 'int (fexcept_t *, int)' (aka 'int (unsigned short *, int)'); ISO C99 and later do not support implicit function declarations}}
  feraiseexcept(FE_INVALID); // expected-note {{include the header <fenv.h> or explicitly provide a declaration for 'feraiseexcept'}} \
   expected-error {{call to undeclared library function 'feraiseexcept' with type 'int (int)'; ISO C99 and later do not support implicit function declarations}}
  fesetexceptflag(flagp, FE_INVALID); // expected-note {{include the header <fenv.h> or explicitly provide a declaration for 'fesetexceptflag'}} \
   expected-error {{call to undeclared library function 'fesetexceptflag' with type 'int (const fexcept_t *, int)' (aka 'int (const unsigned short *, int)'); ISO C99 and later do not support implicit function declarations}}
  fetestexcept(FE_INVALID); // expected-note {{include the header <fenv.h> or explicitly provide a declaration for 'fetestexcept'}} \
   expected-error {{call to undeclared library function 'fetestexcept' with type 'int (int)'; ISO C99 and later do not support implicit function declarations}}
  fegetround(); // expected-note {{include the header <fenv.h> or explicitly provide a declaration for 'fegetround'}} \
   expected-error {{call to undeclared library function 'fegetround' with type 'int (void)'; ISO C99 and later do not support implicit function declarations}}
  fesetround(0); // expected-note {{include the header <fenv.h> or explicitly provide a declaration for 'fesetround'}} \
   expected-error {{call to undeclared library function 'fesetround' with type 'int (int)'; ISO C99 and later do not support implicit function declarations}}
  fegetenv(envp); // expected-note {{include the header <fenv.h> or explicitly provide a declaration for 'fegetenv'}} \
   expected-error {{call to undeclared library function 'fegetenv' with type 'int (fenv_t *)'; ISO C99 and later do not support implicit function declarations}}
  feholdexcept(envp); // expected-note {{include the header <fenv.h> or explicitly provide a declaration for 'feholdexcept'}} \
   expected-error {{call to undeclared library function 'feholdexcept' with type 'int (fenv_t *)'; ISO C99 and later do not support implicit function declarations}}
  fesetenv(envp); // expected-note {{include the header <fenv.h> or explicitly provide a declaration for 'fesetenv'}} \
   expected-error {{call to undeclared library function 'fesetenv' with type 'int (const fenv_t *)'; ISO C99 and later do not support implicit function declarations}}
  feupdateenv(envp); // expected-note {{include the header <fenv.h> or explicitly provide a declaration for 'feupdateenv'}} \
   expected-error {{call to undeclared library function 'feupdateenv' with type 'int (const fenv_t *)'; ISO C99 and later do not support implicit function declarations}}
}
