// RUN: %clang_cc1 -fsyntax-only -verify -pedantic %s

char *funk(int format);
enum Test {A=-1};
char *funk(enum Test x);

int eli(float b); // expected-note {{previous declaration is here}} \
// expected-note{{passing argument to parameter 'b' here}}
int b(int c) {return 1;}

int foo();
int foo() {
  int eli(int (int)); // expected-error {{conflicting types for 'eli'}}
  eli(b); // expected-error{{passing 'int (int)' to parameter of incompatible type 'float'}}
  return 0;
}

int bar();
int bar(int i) // expected-note {{previous definition is here}}
{
  return 0;
}
int bar() // expected-error {{redefinition of 'bar'}} 
{
  return 0;
}

int foobar(int); // note {{previous declaration is here}}
int foobar() // error {{conflicting types for 'foobar'}}
{
  return 0;
}

int wibble(); // expected-note {{previous declaration is here}}
float wibble() // expected-error {{conflicting types for 'wibble'}}
{
  return 0.0f;
}
