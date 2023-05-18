// RUN: %clang_cc1 -fsyntax-only -verify %s

void f()
{
  int x = 0;
  goto label1;

label1: // expected-note{{previous definition is here}}
  x = 1;
  goto label2; // expected-error{{use of undeclared label 'label2'}}

label1: // expected-error{{redefinition of label 'label1'}}
  x = 2;
}

void h()
{
  int x = 0;
  switch (x)
  {
    case 1:;
    default:; // expected-error{{multiple default labels in one switch}}
    default:; // expected-note{{previous case defined here}}
  }
}
