// RUN: %clang_cc1 -fsyntax-only -verify %s

volatile int i;

const int &inc = i++;
const int &dec = i--;

const int &incfail = ++i; // expected-error {{drops 'volatile' qualifier}}
const int &decfail = --i; // expected-error {{drops 'volatile' qualifier}}

// PR7794
void f0(int e) {
  ++(int&)e;
}
