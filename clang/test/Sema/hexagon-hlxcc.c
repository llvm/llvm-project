// RUN: %clang_cc1 -triple hexagon-unknown-elf -target-cpu hexagonv85 \
// RUN:   -target-feature +hlx -fsyntax-only -verify %s

typedef __attribute__((hlxcc)) int invalid_typedef; // expected-warning {{'hlxcc' attribute only applies to functions}}
                                                    // expected-warning@-1 {{'hlxcc' only applies to function types; type here is 'int'}}

void __attribute__((hlxcc(0))) foo0(void); // expected-error {{'hlxcc' attribute takes no arguments}}

void __attribute__((hlxcc, cdecl)) foo1(void);        // expected-error {{not compatible}}
void __attribute__((hlxcc, preserve_hlx)) foo1(void); // expected-error {{not compatible}}

void __attribute__((cdecl)) foo2(void);   // expected-note {{previous declaration is here}}
void __attribute__((hlxcc)) foo2(void) {} // expected-error {{function declared 'hlxcc' here was previously declared 'cdecl'}}

void foo3(void);                          // expected-note {{previous declaration is here}}
void __attribute__((hlxcc)) foo3(void) {} // expected-error {{function declared 'hlxcc' here was previously declared without calling convention}}

void __attribute__((hlxcc)) foo4(void);
int bad = foo4; // expected-error {{incompatible pointer to integer conversion initializing 'int' with an expression of type 'void (void) __attribute__((hlxcc))'}}

// Function pointer and typedef uses are fine.
void (*fp)(int) __attribute__((hlxcc));
typedef void __attribute__((hlxcc)) ft(int);
