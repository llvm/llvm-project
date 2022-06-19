// RUN: %clang_cc1 -fsyntax-only -Wno-strict-prototypes -verify %s

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

void testComplex() {
  // FIXME: Should these be valid?
  _Complex __fp16 a; // expected-error {{'_Complex half' is invalid}}
  __fp16 b;
  a = __builtin_complex(b, b); // expected-error {{'_Complex half' is invalid}}
}
