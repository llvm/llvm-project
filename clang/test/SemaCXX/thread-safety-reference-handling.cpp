// RUN: %clang_cc1 -fsyntax-only -verify -Wthread-safety-analysis -std=c++11 %s
// expected-no-diagnostics

class Base {
public:
  Base() {}
  virtual ~Base();
};

class S : public Base {
public:
  ~S() override = default;
};

void Test() {
  const S &s = S();
}
