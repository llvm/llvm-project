// RUN: %clang_cc1 -triple hexagon-unknown-elf -target-cpu hexagonv85 \
// RUN:   -target-feature +hlx -fsyntax-only -verify %s

typedef short HLX_Vector __attribute__((__vector_size__(2048)));
typedef short HLX_VectorPair __attribute__((__vector_size__(4096)));

typedef __attribute__((preserve_hlx)) int invalid_typedef; // expected-warning {{'preserve_hlx' attribute only applies to functions}}
                                                           // expected-warning@-1 {{'preserve_hlx' only applies to function types; type here is 'int'}}

void __attribute__((preserve_hlx(0))) foo0(void); // expected-error {{'preserve_hlx' attribute takes no arguments}}

void __attribute__((preserve_hlx, cdecl)) foo1(void); // expected-error {{not compatible}}

void __attribute__((cdecl)) foo2(void);          // expected-note {{previous declaration is here}}
void __attribute__((preserve_hlx)) foo2(void) {} // expected-error {{function declared 'preserve_hlx' here was previously declared 'cdecl'}}

void foo3(void);                                 // expected-note {{previous declaration is here}}
void __attribute__((preserve_hlx)) foo3(void) {} // expected-error {{function declared 'preserve_hlx' here was previously declared without calling convention}}

void __attribute__((preserve_hlx)) foo4(void);
int bad = foo4; // expected-error {{incompatible pointer to integer conversion initializing 'int' with an expression of type 'void (void) __attribute__((preserve_hlx))'}}

void __attribute__((preserve_hlx)) ok(int *, HLX_Vector *);

// HLX types are allowed in the signature.
HLX_Vector __attribute__((preserve_hlx)) bar0(void);
HLX_Vector __attribute__((preserve_hlx)) bar1();
void __attribute__((preserve_hlx)) bar2(HLX_Vector a);
void __attribute__((preserve_hlx)) bar3(int, HLX_VectorPair);
void (*__attribute__((preserve_hlx)) bar4)(HLX_Vector);

// Function pointer and typedef uses are fine.
void (*fp)(int) __attribute__((preserve_hlx));
typedef void __attribute__((preserve_hlx)) ft(int);
