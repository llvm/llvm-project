// RUN: %clang_cc1 -fsyntax-only -Wuninitialized -verify %s

void bar(int);
void foo(const int &);

// Test that the warning about self initialization is generated only once.
void test(bool a) {
  int v = v; // expected-warning {{variable 'v' is uninitialized when used within its own initialization}}
  if (a)
    bar(v);
  else
    foo(v);
}
