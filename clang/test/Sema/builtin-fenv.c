// RUN: %clang_cc1 -triple x86_64-unknown-unknown -verify=c,expected -DWRONG_FEXCEPT_T %s -ast-dump | FileCheck %s --check-prefixes=CHECK1,CHECK2
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -verify=c,expected -DRIGHT_FEXCEPT_T %s -ast-dump | FileCheck %s --check-prefixes=CHECK1,CHECK2
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -verify=c,expected -DONLY_FEXCEPT_T %s -ast-dump | FileCheck %s --check-prefixes=CHECK1,CHECK2
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -verify=c,expected -DNO_FEGETEXCEPTFLAG %s -ast-dump 2>&1 | FileCheck %s --check-prefixes=CHECK1

// tests inspired by clang/test/Sema/builtin-setjmp.c

#if WRONG_FEXCEPT_T
typedef unsigned short int fexcept_t;
extern int fegetexceptflag(int, int); // c-warning {{incompatible redeclaration of library function 'fegetexceptflag'}}
                                      // c-note@-1 {{'fegetexceptflag' is a builtin with type 'int (fexcept_t *, int)' (aka 'int (unsigned short *, int)')}}
#elif RIGHT_FEXCEPT_T
// c-no-diagnostics
typedef unsigned short int fexcept_t;
extern int fegetexceptflag(unsigned short int *, int); // OK, right type.
#elif ONLY_FEXCEPT_T
typedef long *fexcept_t;
#endif

void use(void) {
  #pragma STDC FENV_ACCESS ON
  fegetexceptflag(0, 0);
  #if NO_FEGETEXCEPTFLAG
  // cxx-error@-2 {{undeclared identifier 'fegetexceptflag'}}
  // c-error@-3 {{call to undeclared function 'fegetexceptflag'; ISO C99 and later do not support implicit function declarations}}
  // c-warning@-4 {{declaration of built-in function 'fegetexceptflag' requires inclusion of the header <fenv.h>}}
  #elif ONLY_FEXCEPT_T
  // cxx-error@-6 {{undeclared identifier 'fegetexceptflag'}}
  // c-error@-7 {{call to undeclared library function 'fegetexceptflag' with type 'int (fexcept_t *, int)' (aka 'int (long **, int)'); ISO C99 and later do not support implicit function declarations}}
  // c-note@-8 {{include the header <fenv.h> or explicitly provide a declaration for 'fegetexceptflag'}}
  #else
  // cxx-no-diagnostics
  #endif

  #ifdef NO_FEGETEXCEPTFLAG
  // In this case, the regular AST dump doesn't dump the implicit declaration of 'fegetexceptflag'.
  #pragma clang __debug dump fegetexceptflag 
  #endif
}

// CHECK1: FunctionDecl {{.*}} used fegetexceptflag
// CHECK2: BuiltinAttr {{.*}} Implicit
