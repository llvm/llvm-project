// RUN: %clang_cc1 -fsyntax-only -verify %s 
void f();
void f(int);
void f(int, float); 
void f(int, int);
void f(int, ...);

typedef float Float;
void f(int, Float); // expected-note {{previous declaration is here}}

int f(int, Float); // expected-error {{functions that differ only in their return type cannot be overloaded}}

void g(void); // expected-note {{previous declaration is here}}
int g(); // expected-error {{functions that differ only in their return type cannot be overloaded}}

typedef int INT;

class X {
  void f();
  void f(int); // expected-note {{previous declaration is here}}
  void f() const;

  void f(INT); // expected-error{{cannot be redeclared}}

  void g(int); // expected-note {{previous declaration is here}}
  void g(int, float); // expected-note {{previous declaration is here}}
  int g(int, Float); // expected-error {{functions that differ only in their return type cannot be overloaded}}

  static void g(float);
  static void g(int); // expected-error {{static and non-static member functions with the same parameter types cannot be overloaded}}
};
