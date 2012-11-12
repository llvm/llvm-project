// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11
// expected-no-diagnostics

// 13.3.3.2 Ranking implicit conversion sequences
// conversion of A::* to B::* is better than conversion of A::* to C::*,
struct A {
int Ai;
}; 

struct B : public A {}; 
struct C : public B {}; 

const char * f(int C::*){ return ""; } 
int f(int B::*) { return 1; } 

struct D : public C {}; 

const char * g(int B::*){ return ""; } 
int g(int D::*) { return 1; } 

void test() 
{
  int i = f(&A::Ai);

  const char * str = g(&A::Ai);
}

// conversion of B::* to C::* is better than conversion of A::* to C::*
typedef void (A::*pmfa)();
typedef void (B::*pmfb)();
typedef void (C::*pmfc)();

struct X {
	operator pmfa();
	operator pmfb();
};


void g(pmfc);

void test2(X x) 
{
    g(x);
}

