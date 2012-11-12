// RUN: %clang_cc1 %s -verify -fsyntax-only

void foo(void);
void foo(void) {} 
void foo(void);
void foo(void); // expected-note {{previous declaration is here}}

void foo(int); // expected-error {{conflicting types for 'foo'}}

int funcdef()
{
 return 0;
}

int funcdef();

int funcdef2() { return 0; } // expected-note {{previous definition is here}}
int funcdef2() { return 0; } // expected-error {{redefinition of 'funcdef2'}}

// PR2502
void (*f)(void);
void (*f)() = 0;

typedef __attribute__(( ext_vector_type(2) )) int Vi2;
typedef __attribute__(( ext_vector_type(2) )) float Vf2;

Vf2 g0; // expected-note {{previous definition is here}}
Vi2 g0; // expected-error {{redefinition of 'g0'}}

_Complex int g1; // expected-note {{previous definition is here}}
_Complex float g1; // expected-error {{redefinition of 'g1'}}

// rdar://6096412
extern char i6096412[10];
extern char i6096412[];
void foo6096412(void) {
  int x = sizeof(i6096412);
}

