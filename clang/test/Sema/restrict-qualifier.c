// RUN: %clang_cc1 -std=c2y -fsyntax-only -verify -pedantic %s

typedef int (*T1)[2];
restrict T1 t1;
static_assert(_Generic(typeof (t1), int (*restrict)[2] : 1, default : 0));

typedef int *T2[2];
restrict T2 t2;
static_assert(_Generic(typeof (t2), int *restrict[2] : 1, default : 0));

typedef int *T3[2][2];
restrict T3 t3;
static_assert(_Generic(typeof (t3), int *restrict[2][2] : 1, default : 0));
static_assert(_Generic(void(T3 restrict), void(int *restrict (*)[2]): 1, default: 0));

typedef int (*t4)();
typedef t4 t5[2];
typedef t5 restrict t6; // expected-error {{pointer to function type 'int (void)' may not be 'restrict' qualified}}

typedef int t7[2];
typedef t7 restrict t8; // expected-error {{restrict requires a pointer or reference ('t7' (aka 'int[2]')}}
