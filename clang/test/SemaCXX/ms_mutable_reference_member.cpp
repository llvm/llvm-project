// RUN: %clang_cc1 %s -fsyntax-only -verify -fms-compatibility

struct S {
  mutable int &a; // expected-warning {{'mutable' on a reference type is a Microsoft extension}}
  S(int &b) : a(b) {}
};

int main() {
  int a = 0;
  const S s(a);
  s.a = 10;
  return s.a + a;
}
