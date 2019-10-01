// RUN: %clang_cc1 -fsyntax-only -verify %s -pedantic -std=c99

int __attribute__(()) x;

__inline void __attribute__((__always_inline__, __nodebug__))
foo(void) {
}


__attribute__(()) y;   // expected-warning {{defaults to 'int'}}

// PR2796
int (__attribute__(()) *z)(long y);


void f1(__attribute__(()) int x);

int f2(y, __attribute__(()) x);     // expected-error {{expected identifier}}

// This is parsed as a normal argument list (with two args that are implicit
// int) because the __attribute__ is a declspec.
void f3(__attribute__(()) x,  // expected-warning {{defaults to 'int'}}
        y);               // expected-warning {{defaults to 'int'}}

void f4(__attribute__(()));   // expected-error {{expected parameter declarator}}


// This is ok, the __attribute__ applies to the pointer.
int baz(int (__attribute__(()) *x)(long y));

void g1(void (*f1)(__attribute__(()) int x));
void g2(int (*f2)(y, __attribute__(()) x));    // expected-error {{expected identifier}}
void g3(void (*f3)(__attribute__(()) x, int y));  // expected-warning {{defaults to 'int'}}
void g4(void (*f4)(__attribute__(())));  // expected-error {{expected parameter declarator}}


void (*h1)(void (*f1)(__attribute__(()) int x));
void (*h2)(int (*f2)(y, __attribute__(()) x));    // expected-error {{expected identifier}}

void (*h3)(void (*f3)(__attribute__(()) x));   // expected-warning {{defaults to 'int'}}
void (*h4)(void (*f4)(__attribute__(())));  // expected-error {{expected parameter declarator}}



// rdar://6131260
int foo42(void) {
  int x, __attribute__((unused)) y, z;
  return 0;
}

// rdar://6096491
void __attribute__((noreturn)) d0(void), __attribute__((noreturn)) d1(void);

void d2(void) __attribute__((noreturn)), d3(void) __attribute__((noreturn));


// PR6287
void __attribute__((returns_twice)) returns_twice_test();

int aligned(int);
int __attribute__((vec_type_hint(char, aligned(16) )) missing_rparen_1; // expected-error 2{{expected ')'}} expected-note {{to match}} expected-warning {{does not declare anything}}
int __attribute__((mode(x aligned(16) )) missing_rparen_2; // expected-error 2{{expected ')'}}
int __attribute__((format(printf, 0 aligned(16) )) missing_rparen_3; // expected-error 2{{expected ')'}}



int testFundef1(int *a) __attribute__((nonnull(1))) { // \
    // expected-warning {{GCC does not allow 'nonnull' attribute in this position on a function definition}}
  return *a;
}

// noreturn is lifted to type qualifier
void testFundef2() __attribute__((noreturn)) { // \
    // expected-warning {{GCC does not allow 'noreturn' attribute in this position on a function definition}}
  testFundef2();
}

int testFundef3(int *a) __attribute__((nonnull(1), // \
    // expected-warning {{GCC does not allow 'nonnull' attribute in this position on a function definition}}
                                     pure)) { // \
    // expected-warning {{GCC does not allow 'pure' attribute in this position on a function definition}}
  return *a;
}

int testFundef4(int *a) __attribute__((nonnull(1))) // \
    // expected-warning {{GCC does not allow 'nonnull' attribute in this position on a function definition}}
                      __attribute((pure)) { // \
    // expected-warning {{GCC does not allow 'pure' attribute in this position on a function definition}}
  return *a;
}

// GCC allows these
void testFundef5() __attribute__(()) { }

__attribute__((pure)) int testFundef6(int a) { return a; }

void deprecatedTestFun(void) __attribute__((deprecated()));

struct s {
  int a;
};

// This test ensure compatibility with parsing GNU-style attributes
// where the attribute is on a separate line from the elaborated type
// specifier.
struct s
__attribute__((used)) bar;

// Ensure that attributes must be separated by a comma (PR38352).
__attribute__((const const)) int PR38352(void); // expected-error {{expected ')'}}
// Also ensure that we accept spurious commas.
__attribute__((,,,const)) int PR38352_1(void);
__attribute__((const,,,)) int PR38352_2(void);
__attribute__((const,,,const)) int PR38352_3(void);
__attribute__((,,,const,,,const,,,)) int PR38352_4(void);
