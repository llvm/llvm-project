// RUN: %clangxx_tysan -O0 %s -o %t && %run %t

struct A {
  int elems[3];
};

A a;

int main() { a.elems[0] = 1; }
