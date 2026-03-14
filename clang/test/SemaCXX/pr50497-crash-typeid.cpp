// RUN: %clang_cc1 -verify %s -Wno-unevaluated-expression
// Don't crash (PR50497).

// expected-no-diagnostics
namespace std {
class type_info;
}

class Ex {
  // polymorphic
  virtual ~Ex();
};
void Frob(const std::type_info &type);

void Foo(Ex *ex) {
  // generic lambda
  [=](auto rate) {
    // typeid
    Frob(typeid(*ex));
  }(1);

  [=](auto rate) {
    // unevaluated nested typeid
    Frob(typeid((typeid(*ex), ex)));
  }(1);
}
