// RUN: %clang_cc1 %s -std=c11 -fsyntax-only -verify=compat -Wpre-c11-compat
// RUN: %clang_cc1 %s -std=c99 -fsyntax-only -verify=ext -pedantic
// RUN: %clang_cc1 %s -std=c11 -fsyntax-only -verify=good
// RUN: %clang_cc1 -x c++ %s -fsyntax-only -verify=ext -pedantic

// good-no-diagnostics

extern _Noreturn void exit(int); /* compat-warning {{'_Noreturn' is incompatible with C standards before C11}}
                                    ext-warning {{'_Noreturn' is a C11 extension}}
                                  */

void func(void) {
  static _Thread_local int tl;   /* compat-warning {{'_Thread_local' is incompatible with C standards before C11}}
                                    ext-warning {{'_Thread_local' is a C11 extension}}
                                  */
  _Alignas(8) char c;            /* compat-warning {{'_Alignas' is incompatible with C standards before C11}}
                                    ext-warning {{'_Alignas' is a C11 extension}}
                                  */
  _Atomic int i1;                /* compat-warning {{'_Atomic' is incompatible with C standards before C11}}
                                    ext-warning {{'_Atomic' is a C11 extension}}
                                  */
  _Atomic(int) i2;               /* compat-warning {{'_Atomic' is incompatible with C standards before C11}}
                                    ext-warning {{'_Atomic' is a C11 extension}}
                                  */

  _Static_assert(1, "");         /* compat-warning {{'_Static_assert' is incompatible with C standards before C11}}
                                    ext-warning {{'_Static_assert' is a C11 extension}}
                                  */

  (void)_Generic(1, int : 1);    /* compat-warning {{'_Generic' is incompatible with C standards before C11}}
                                    ext-warning {{'_Generic' is a C11 extension}}
                                  */
  (void)_Alignof(int);           /* compat-warning {{'_Alignof' is incompatible with C standards before C11}}
                                    ext-warning {{'_Alignof' is a C11 extension}}
                                  */
}

