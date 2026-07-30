// RUN: %clang_cc1 -fsyntax-only -verify %s
class C { };

C c;

void D(int);

class D {};

void foo()
{
  D(5);
  class D d;
}

class D; // expected-note {{previous use is here}}

enum D; // expected-error {{use of 'D' with tag type that does not match previous declaration}}

class A * A;

class A * a2;

void bar()
{
  A = 0;
}

void C(int);

void bar2()
{
  C(17);
}

extern int B;
class B;
class B {};
int B;

enum E { e1_val };
E e1;

void E(int);

void bar3() {
  E(17);
}

enum E e2;

enum E2 { E2 };
