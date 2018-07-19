// RUN: %clang_cc1 -fsyntax-only -verify -fno-rtti %s

namespace std {
  class type_info;
}

void f()
{
  (void)typeid(int); // expected-error {{cannot use typeid with -fno-rtti}}
}

namespace {
struct A {
  virtual ~A(){};
};

struct B : public A {
  B() : A() {}
};
}

bool isa_B(A *a) {
  return dynamic_cast<B *>(a) != 0; // expected-error {{cannot use dynamic_cast with -fno-rtti}}
}

void* getMostDerived(A* a) {
  // This cast does not use RTTI.
  return dynamic_cast<void *>(a);
}
