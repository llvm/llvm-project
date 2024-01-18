// RUN: %clang_cc1 -x c -verify %s
// RUN: %clang_cc1 -x c -verify %s -ffixed-point -DFIXED_POINT=1

int _Accum;

#ifdef FIXED_POINT
// expected-error@4{{cannot combine with previous 'int' declaration specifier}}
// expected-warning@4{{declaration does not declare anything}}
#else
// expected-no-diagnostics
#endif
