// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: cp %s %t.cpp
// RUN: not %clang_cc1 %t.cpp -fixit
// RUN: %clang_cc1 -fsyntax-only %t.cpp

void test1(int a;) { // expected-error{{unexpected ';' before ')'}}
  while (a > 5;) {} // expected-error{{unexpected ';' before ')'}}
  for (int c  = 0; c < 21; ++c;) {} // expected-error{{unexpected ';' before ')'}}
  int d = int(3 + 4;); // expected-error{{unexpected ';' before ')'}}
  int e[5;]; // expected-error{{unexpected ';' before ']'}}
  e[a+1;] = 4; // expected-error{{unexpected ';' before ']'}}
  int f[] = {1,2,3;}; // expected-error{{unexpected ';' before '}'}}
}
