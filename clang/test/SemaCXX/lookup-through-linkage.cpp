// RUN: %clang_cc1 %s -verify

// expected-no-diagnostics

extern "C++" {
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

extern "C" {
extern "C++" {
namespace D {
namespace E {
int bar;
}
} // namespace A
namespace F {
void foo() {
  using namespace D;
  (void)E::bar;
}
} // namespace C
}
}
