// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64-unknown-linux-gnu %s

// Correct cases.
typedef int __attribute__((mode(byte))) __attribute__((vector_size(256))) vec_t1;
typedef int __attribute__((mode(QI))) __attribute__((vector_size(256))) vec_t2;
typedef int __attribute__((mode(SI))) __attribute__((vector_size(256))) vec_t3;
typedef int __attribute__((mode(DI))) __attribute__((vector_size(256)))vec_t4;
typedef float __attribute__((mode(SF))) __attribute__((vector_size(256))) vec_t5;
typedef float __attribute__((mode(DF))) __attribute__((vector_size(256))) vec_t6;
typedef float __attribute__((mode(XF))) __attribute__((vector_size(256))) vec_t7;

typedef int v8qi __attribute__ ((mode(QI))) __attribute__ ((vector_size(8)));
typedef int v8qi __attribute__ ((mode(V8QI)));
// expected-warning@-1{{specifying vector types with the 'mode' attribute is deprecated; use the 'vector_size' attribute instead}}

typedef float v4sf __attribute__((mode(V4SF)));
// expected-warning@-1{{specifying vector types with the 'mode' attribute is deprecated; use the 'vector_size' attribute instead}}
typedef float v4sf __attribute__((mode(SF)))  __attribute__ ((vector_size(16)));

// Incorrect cases.
typedef float __attribute__((mode(QC))) __attribute__((vector_size(256))) vec_t8;
// expected-error@-1{{unsupported machine mode 'QC'}}
// expected-error@-2{{type of machine mode does not match type of base type}}
typedef _Complex float __attribute__((mode(HC))) __attribute__((vector_size(256))) vec_t9;
// expected-error@-1{{invalid vector element type '_Complex float'}}
typedef int __attribute__((mode(SC))) __attribute__((vector_size(256))) vec_t10;
// expected-error@-1{{type of machine mode does not match type of base type}}
// expected-error@-2{{type of machine mode does not support base vector types}}
typedef float __attribute__((mode(DC))) __attribute__((vector_size(256))) vec_t11;
// expected-error@-1{{type of machine mode does not match type of base type}}
// expected-error@-2{{type of machine mode does not support base vector types}}
typedef _Complex float __attribute__((mode(XC))) __attribute__((vector_size(256))) vec_t12;
// expected-error@-1{{invalid vector element type '_Complex float'}}
typedef int __attribute__((mode(V3QI))) v3qi;
// expected-error@-1{{unknown machine mode 'V3QI'}}
