// RUN: %clang_cc1 -fsyntax-only -verify %s

double operator +(double, double); // expected-error{{overloaded 'operator+' must have at least one parameter of class or enumeration type}}

struct A
{
  operator int();
};

int main()
{
  A a, b;
  int i0 = a + 1;
  int i1 = a + b;
}
