// RUN: %clang_cc1 -std=c++20 %s -verify

// expected-no-diagnostics
export module X;
export {
  namespace A {
  namespace B {
  int bar;
  }
  } // namespace A
  namespace C {
  void foo() {
    using namespace A;
    (void)B::bar;
  }
  } // namespace C
}

export {
  namespace D {
  namespace E {
  int bar;
  }
  } // namespace D
  namespace F {
  void foo() {
    using namespace D;
    (void)E::bar;
  }
  } // namespace F
}
