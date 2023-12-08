// RUN: %clang_cc1 -fsyntax-only %s -verify

struct Data {
  char *a;
  char *b;
  bool *c;
};

int main() {
  Data in;
  in.a = new char[](); // expected-error {{cannot determine allocated array size from initializer}}
  in.c = new bool[100]();
  in.b = new char[100]();
}
