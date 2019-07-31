// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fsyntax-only -verify -DNO_JMP_BUF %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fsyntax-only -verify %s

#ifdef NO_JMP_BUF
extern long setjmp(long *);   // expected-warning {{declaration of built-in function 'setjmp' requires the declaration of the 'jmp_buf' type, commonly provided in the header <setjmp.h>.}}
#else
typedef long jmp_buf;
extern int setjmp(char);      // expected-warning@8 {{incompatible redeclaration of library function 'setjmp'}}
                              // expected-note@8    {{'setjmp' is a builtin with type 'int (jmp_buf)' (aka 'int (long)')}}
#endif
