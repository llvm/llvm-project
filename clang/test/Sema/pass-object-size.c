// RUN: %clang_cc1 -fsyntax-only -verify %s -triple x86_64-linux-gnu -Wincompatible-pointer-types
//
// Tests for the pass_object_size attribute
// Non-failure cases are covered in test/CodeGen/pass-object-size.c

void a(void *p __attribute__((pass_object_size))); //expected-error{{'pass_object_size' attribute takes one argument}}
void b(void *p __attribute__((pass_object_size(1.0)))); //expected-error{{'pass_object_size' attribute requires parameter 1 to be an integer constant}}

void c(void *p __attribute__((pass_object_size(4)))); //expected-error{{'pass_object_size' attribute requires integer constant between 0 and 3 inclusive}}
void d(void *p __attribute__((pass_object_size(-1)))); //expected-error{{'pass_object_size' attribute requires integer constant between 0 and 3 inclusive}}

void e(void *p __attribute__((pass_object_size(1ULL<<32)))); //expected-error{{integer constant expression evaluates to value 4294967296 that cannot be represented in a 32-bit unsigned integer type}}

void f(char p __attribute__((pass_object_size(0)))); //expected-error{{'pass_object_size' attribute only applies to constant pointer arguments}}
void g(const char p __attribute__((pass_object_size(0)))); //expected-error{{'pass_object_size' attribute only applies to constant pointer arguments}}
void h(char *p __attribute__((pass_object_size(0)))) {} //expected-error{{pass_object_size attribute only applies to constant pointer arguments}}
void i(char *p __attribute__((pass_object_size(0)))); // OK -- const is only necessary on definitions, not decls.
void j(char *p __attribute__((pass_object_size(0), pass_object_size(1)))); //expected-error{{'pass_object_size' attribute can only be applied once per parameter}}

#define PS(N) __attribute__((pass_object_size(N)))
#define overloaded __attribute__((overloadable))
void Overloaded(void *p PS(0)) overloaded; //expected-note{{previous declaration is here}}
void Overloaded(void *p PS(1)) overloaded; //expected-error{{conflicting pass_object_size attributes on parameters}}
void Overloaded2(void *p PS(1), void *p2 PS(0)) overloaded; //expected-note{{previous declaration is here}}
void Overloaded2(void *p PS(0), void *p2 PS(1)) overloaded; //expected-error{{conflicting pass_object_size attributes on parameters}}

void Overloaded3(void *p PS(0), void *p2) overloaded; //expected-note{{previous declaration is here}}
void Overloaded3(void *p, void *p2 PS(0)) overloaded; //expected-error{{conflicting pass_object_size attributes on parameters}}

void TakeFn(void (*)(void *));
void TakeFnOvl(void (*)(void *)) overloaded;
void TakeFnOvl(void (*)(int *)) overloaded;

void NotOverloaded(void *p PS(0));
void IsOverloaded(void *p PS(0)) overloaded;
void IsOverloaded(char *p) overloaded; // char* inestead of void* is intentional
void FunctionPtrs() {
  void (*p)(void *) = NotOverloaded; //expected-error{{cannot take address of function 'NotOverloaded' because parameter 1 has pass_object_size attribute}}
  void (*p2)(void *) = &NotOverloaded; //expected-error{{cannot take address of function 'NotOverloaded' because parameter 1 has pass_object_size attribute}}

  void (*p3)(void *) = IsOverloaded; //expected-warning{{incompatible pointer types initializing 'void (*)(void *)' with an expression of type '<overloaded function type>'}} expected-note@-6{{candidate address cannot be taken because parameter 1 has pass_object_size attribute}} expected-note@-5{{type mismatch}}
  void (*p4)(void *) = &IsOverloaded; //expected-warning{{incompatible pointer types initializing 'void (*)(void *)' with an expression of type '<overloaded function type>'}} expected-note@-7{{candidate address cannot be taken because parameter 1 has pass_object_size attribute}} expected-note@-6{{type mismatch}}

  void (*p5)(char *) = IsOverloaded;
  void (*p6)(char *) = &IsOverloaded;

  TakeFn(NotOverloaded); //expected-error{{cannot take address of function 'NotOverloaded' because parameter 1 has pass_object_size attribute}}
  TakeFn(&NotOverloaded); //expected-error{{cannot take address of function 'NotOverloaded' because parameter 1 has pass_object_size attribute}}

  TakeFnOvl(NotOverloaded); //expected-error{{cannot take address of function 'NotOverloaded' because parameter 1 has pass_object_size attribute}}
  TakeFnOvl(&NotOverloaded); //expected-error{{cannot take address of function 'NotOverloaded' because parameter 1 has pass_object_size attribute}}

  int P;
  (&NotOverloaded)(&P); //expected-error{{cannot take address of function 'NotOverloaded' because parameter 1 has pass_object_size attribute}}
  (&IsOverloaded)(&P); //expected-error{{no matching function}} expected-note@35{{candidate address cannot be taken because parameter 1 has pass_object_size attribute}} expected-note@36{{candidate function not viable: no known conversion from 'int *' to 'char *' for 1st argument}}
}
