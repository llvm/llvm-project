// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace std {
  template<typename T> class vector { }; // expected-note{{candidate}}
}

typedef int INT;
typedef float Real;

void test() {
  using namespace std;

  std::vector<INT> v1;
  vector<Real> v2;
  v1 = v2; // expected-error{{no viable overloaded '='}}
}
