// RUN: %clang_cc1 -fsyntax-only -verify %s

// PR6647
class C {
  // After the error, the rest of the tokens inside the default arg should be
  // skipped, avoiding a "expected ';' after class" after 'undecl'.
  void m(int x = undecl + 0); // expected-error {{use of undeclared identifier 'undecl'}}
};

typedef struct Inst {
  void m(int x=0);
} *InstPtr;

struct X {
  void f(int x = 1:); // expected-error {{unexpected end of default argument expression}}
};

// PR13657
struct T {
  template <typename A, typename B> struct T1 { enum {V};};
  template <int A, int B> struct T2 { enum {V}; };
  template <int, int> static int func(int);


  void f1(T1<int, int> = T1<int, int>());
  void f2(T1<int, double> = T1<int, double>(), T2<0, 5> = T2<0, 5>());
  void f3(int a = T2<0, (T1<int, int>::V > 10) ? 5 : 6>::V, bool b = 4<5 );
  void f4(bool a = 1 < 0, bool b = 2 > 0 );
  void f5(bool a = 1 > T2<0, 0>::V, bool b = T1<int,int>::V < 3, int c = 0);
  void f6(bool a = T2<0,3>::V < 4, bool b = 4 > T2<0,3>::V);
  void f7(bool a = T1<int, bool>::V < 3);
  void f8(int = func<0,1<2>(0), int = 1<0, T1<int,int>(int) = 0);
};

// rdar://18508589
struct S { 
  void f(int &r = error);  // expected-error {{use of undeclared identifier 'error'}}
};

struct U {
  void i(int x = ) {} // expected-error{{expected expression}}
  typedef int *fp(int x = ); // expected-error{{default arguments can only be specified for parameters in a function declaration}}
};
