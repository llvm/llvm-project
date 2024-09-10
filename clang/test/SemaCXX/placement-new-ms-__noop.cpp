// RUN: %clang_cc1 -fsyntax-only -fms-extensions -verify %s -std=c++11
// expected-no-diagnostics

struct S {
  void* operator new(__SIZE_TYPE__, int);
};

int main() {
  // MSVC supports __noop with no arguments or (), so we do as well.
  new (__noop) S;
  new ((__noop)) S;
}
