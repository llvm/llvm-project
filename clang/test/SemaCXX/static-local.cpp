// RUN: %clang_cc1 -fsyntax-only -std=c++20 -verify %s

int main() {
  static struct StaticLocal { // expected-error {{templates cannot be declared inside of a local class}}
    void bar(auto x) {}
  } s;
  (void)s;
}
