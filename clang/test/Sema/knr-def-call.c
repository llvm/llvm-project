// RUN: %clang_cc1 -triple i386-pc-unknown -Wconversion -Wliteral-conversion -fsyntax-only -verify %s

// C DR #316, PR 3626.
void f0(a, b, c, d) int a,b,c,d; {}
void t0(void) {
  f0(1);  // expected-warning{{too few arguments}}
}

void f1(a, b) int a, b; {}
void t1(void) {
  f1(1, 2, 3); // expected-warning{{too many arguments}}
}

void f2(float); // expected-note{{previous declaration is here}}
void f2(x) float x; { } // expected-warning{{promoted type 'double' of K&R function parameter is not compatible with the parameter type 'float' declared in a previous prototype}}

typedef void (*f3)(void);
f3 t3(int b) { return b? f0 : f1; } // okay

// <rdar://problem/8193107>
void f4() {
    char *rindex();
}

char *rindex(s, c)
     register char *s, c; // expected-warning{{promoted type 'char *' of K&R function parameter is not compatible with the parameter type 'const char *' declared in a previous prototype}}
{
  return 0;
}

// PR8314
void proto(int);
void proto(x)
     int x;
{
}

void use_proto() {
  proto(42.1); // expected-warning{{implicit conversion from 'double' to 'int' changes value from 42.1 to 42}}
  (&proto)(42.1); // expected-warning{{implicit conversion from 'double' to 'int' changes value from 42.1 to 42}}
}

// PR31020
void func(short d) __attribute__((cdecl)); // expected-note{{previous declaration is here}}
void __attribute__((cdecl)) func(d)
  short d; // expected-warning{{promoted type 'int' of K&R function parameter is not compatible with the parameter type 'short' declared in a previous prototype}}
{}
