// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited %s -o /dev/null
// PR 7104

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
