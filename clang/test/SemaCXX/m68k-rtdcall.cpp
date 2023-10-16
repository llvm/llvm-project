// RUN: %clang_cc1 -triple m68k-linux-gnu -fsyntax-only %s

class A {
public:
  void __attribute__((m68k_rtd)) member() {}
};

void test() {
  A a;
  a.member();

  auto f = [](int b) __attribute__((m68k_rtd)) {};
  f(87);
};
