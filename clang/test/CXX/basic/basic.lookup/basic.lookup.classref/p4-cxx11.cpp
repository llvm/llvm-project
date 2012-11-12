// RUN: %clang_cc1 -std=c++11 %s -verify
// expected-no-diagnostics

struct A { void f(); };
struct C { void f(); };
struct B : A { typedef A X; };
struct D : C { typedef C X;   void g(); };

void D::g() 
{
    B * b = new B;
    b->X::f(); // lookup for X finds B::X
}

typedef int X;
void h(void) 
{
    B * b = new B;
    b->X::f(); // lookup for X finds B::X
}


