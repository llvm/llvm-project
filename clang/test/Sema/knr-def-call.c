// RUN: %clang_cc1 -triple i386-pc-unknown -Wconversion -Wliteral-conversion -fsyntax-only -verify %s

// C DR #316, PR 3626.
void f0(a, b, c, d) int a,b,c,d; {} // expected-warning {{a function definition without a prototype is deprecated in all versions of C and is not supported in C23}}
void t0(void) {
  f0(1);  // expected-warning{{too few arguments}}
}

void f1(a, b) int a, b; {} // expected-warning {{a function definition without a prototype is deprecated in all versions of C and is not supported in C23}}
void t1(void) {
  f1(1, 2, 3); // expected-warning{{too many arguments}}
}

void f2(float); // expected-note{{previous declaration is here}}
void f2(x) float x; { } // expected-warning{{promoted type 'double' of K&R function parameter is not compatible with the parameter type 'float' declared in a previous prototype}} \
                           expected-warning {{a function definition without a prototype is deprecated in all versions of C and is not supported in C23}}

typedef void (*f3)(void);
f3 t3(int b) { return b? f0 : f1; } // okay

void f4() {
    char *rindex();
}

// FIXME: we should diagnose this case, but when merging function declarations,
// we don't diagnose it because the function is a builtin.
char *rindex(s, c)
     register char *s, c; // expected-warning{{promoted type 'char *' of K&R function parameter is not compatible with the parameter type 'const char *' declared in a previous prototype}}
{
  return 0;
}

// PR8314
void proto(int);
void proto(x) // expected-warning {{a function definition without a prototype is deprecated in all versions of C and is not supported in C23}}
     int x;
{
}

void use_proto() {
  proto(42.1); // expected-warning{{implicit conversion from 'double' to 'int' changes value from 42.1 to 42}}
  (&proto)(42.1); // expected-warning{{implicit conversion from 'double' to 'int' changes value from 42.1 to 42}}
}

// PR31020
void func(short d) __attribute__((cdecl)); // expected-note{{previous declaration is here}}
void __attribute__((cdecl)) func(d) // expected-warning {{a function definition without a prototype is deprecated in all versions of C and is not supported in C23}}
  short d; // expected-warning{{promoted type 'int' of K&R function parameter is not compatible with the parameter type 'short' declared in a previous prototype}}
{}
