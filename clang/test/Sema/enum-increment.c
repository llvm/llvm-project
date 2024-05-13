// RUN: %clang_cc1 -fsyntax-only %s -verify
// expected-no-diagnostics
enum A { A1, A2, A3 };
typedef enum A A;
void test(void) {
  A a;
  a++;
  a--;
  ++a;
  --a;
  a = a + 1;
  a = a - 1;
}
