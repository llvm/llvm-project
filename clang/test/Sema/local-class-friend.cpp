// RUN: %clang_cc1 -verify -fsyntax-only %s
// expected-no-diagnostics

void foo()
{ class c1 {
    private:
      int testVar;
    public:
      friend class c2;
  };

  class c2 {
    void f(c1 obj) {
      int a = obj.testVar; // Ok
    }
  };
}
