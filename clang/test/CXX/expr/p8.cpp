// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11

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

  using IA = int[];
  void(+IA{ 1, 2, 3 }); // expected-error {{array prvalue}}
  void(*IA{ 1, 2, 3 }); // expected-error {{array prvalue}}
  void(IA{ 1, 2, 3 } + 0); // expected-error {{array prvalue}}
  void(IA{ 1, 2, 3 } - 0); // expected-error {{array prvalue}}
  void(0 + IA{ 1, 2, 3 }); // expected-error {{array prvalue}}
  void(0 - IA{ 1, 2, 3 }); // expected-error {{array prvalue}}
  void(IA{ 1, 2, 3 } - IA{ 1, 2, 3 }); // expected-error {{array prvalue}}
}
