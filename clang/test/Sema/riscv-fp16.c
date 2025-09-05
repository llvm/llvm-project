// RUN: %clang_cc1 -fsyntax-only -triple riscv32 -Wno-strict-prototypes -verify %s
// RUN: %clang_cc1 -fsyntax-only -triple riscv64 -Wno-strict-prototypes -verify %s
// REQUIRES: riscv-registered-target

// Functions cannot have parameters of type __fp16.
extern void f (__fp16); // expected-error {{parameters cannot have __fp16 type; did you forget * ?}}
extern void g (__fp16 *);

extern void (*pf) (__fp16);  // expected-error {{parameters cannot have __fp16 type; did you forget * ?}}
extern void (*pg) (__fp16*);

typedef void(*tf) (__fp16);  // expected-error {{parameters cannot have __fp16 type; did you forget * ?}}
typedef void(*tg) (__fp16*);

void kf(a)
 __fp16 a; {  // expected-error {{parameters cannot have __fp16 type; did you forget * ?}}
}

void kg(a)
 __fp16 *a; {
}

// Functions cannot return type __fp16.
extern __fp16 f1 (void); // expected-error {{function return value cannot have __fp16 type; did you forget * ?}}
extern __fp16 *g1 (void);

extern __fp16 (*pf1) (void); // expected-error {{function return value cannot have __fp16 type; did you forget * ?}}
extern __fp16 *(*gf1) (void);

typedef __fp16 (*tf1) (void); // expected-error {{function return value cannot have __fp16 type; did you forget * ?}}
typedef __fp16 *(*tg1) (void);
