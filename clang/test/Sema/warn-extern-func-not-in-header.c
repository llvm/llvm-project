// RUN: %clang_cc1 -fsyntax-only -verify -Wexternal-declaration %s

extern int
foo(int); // expected-warning{{extern function 'foo' declared in main file}}

int bar(int);
int bar(int x) {
  return x + 1;
}

int main() {
  return foo(42) + bar(10);
}