// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

int a0;
const volatile int a1 = 2;
int a2[16];
int a3();

void f0(int);
void f1(int *);
void f2(int (*)());

int main()
{
  f0(a0);
  f0(a1);
  f1(a2);
  f2(a3);
}
