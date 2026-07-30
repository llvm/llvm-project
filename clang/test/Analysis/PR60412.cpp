// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.deadcode.UnreachableCode -verify %s
// expected-no-diagnostics

struct Test {
  Test() {}
  ~Test();
};

int foo() {
  struct a {
    Test b, c;
  } d;
  return 1;
}

int main() {
  if (foo()) return 1; // <- this used to warn as unreachable
}
