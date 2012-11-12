// RUN: %clang_cc1 -std=gnu89 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify %s

#if __STDC_VERSION__ >= 199901
#define GNU_INLINE __attribute((__gnu_inline__))
#else
#define GNU_INLINE
#endif

// PR5253
// rdar://9559708 (same extension in C99 mode)
// GNU Extension: check that we can redefine an extern inline function
GNU_INLINE extern inline int f(int a) {return a;}
int f(int b) {return b;} // expected-note{{previous definition is here}}
// And now check that we can't redefine a normal function
int f(int c) {return c;} // expected-error{{redefinition of 'f'}}

// Check that we can redefine an extern inline function as a static function
GNU_INLINE extern inline int g(int a) {return a;}
static int g(int b) {return b;}

// Check that we ensure the types of the two definitions are the same
GNU_INLINE extern inline int h(int a) {return a;} // expected-note{{previous definition is here}}
int h(short b) {return b;}  // expected-error{{conflicting types for 'h'}}
