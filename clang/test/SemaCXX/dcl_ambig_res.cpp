// RUN: %clang_cc1 -fsyntax-only -pedantic -verify %s

// PR13819
// REQUIRES: LP64

// [dcl.ambig.res]p1:
struct S { 
  S(int); 
  void bar();
}; 

int returns_an_int();

void foo(double a) 
{ 
  S w(int(a)); // expected-warning{{disambiguated as a function declaration}} expected-note{{add a pair of parentheses}} 
  w(17);
  S x1(int()); // expected-warning{{disambiguated as a function declaration}} expected-note{{add a pair of parentheses}} 
  x1(&returns_an_int);
  S y((int)a); 
  y.bar();
  S z = int(a);
  z.bar();
} 

// [dcl.ambig.res]p3:
char *p; 
void *operator new(__SIZE_TYPE__, int); 
void foo3() { 
  const int x = 63; 
  new (int(*p)) int; //new-placement expression 
  new (int(*[x])); //new type-id 
} 

// [dcl.ambig.res]p4:
template <class T>  // expected-note{{here}}
struct S4 { 
  T *p; 
}; 
S4<int()> x; //type-id 
S4<int(1)> y; // expected-error{{must be a type}}

// [dcl.ambig.res]p5:
void foo5() 
{ 
  (void)sizeof(int(1)); //expression 
  (void)sizeof(int()); // expected-error{{function type}}
}

// [dcl.ambig.res]p6:
void foo6() 
{ 
  (void)(int(1)); //expression 
  (void)(int())1; // expected-error{{to 'int ()'}}
} 

// [dcl.ambig.res]p7:
class C7 { }; 
void f7(int(C7)) { } // expected-note{{candidate}}
int g7(C7); 
void foo7() { 
  f7(1); // expected-error{{no matching function}}
  f7(g7); //OK 
} 

void h7(int *(C7[10])) { } // expected-note{{previous}}
void h7(int *(*_fp)(C7 _parm[10])) { } // expected-error{{redefinition}}

struct S5 {
  static bool const value = false;
};
int foo8() {
  int v(int(S5::value)); // expected-warning{{disambiguated as a function declaration}} expected-note{{add a pair of parentheses}} expected-error{{parameter declarator cannot be qualified}}
}

template<typename T>
void rdar8739801( void (T::*)( void ) __attribute__((unused)) );
