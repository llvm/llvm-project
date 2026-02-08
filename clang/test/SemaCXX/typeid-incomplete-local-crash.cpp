// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify %s

namespace std {
class type_info;
}

namespace gh176397 {
auto recursiveLambda = [](auto, int) {
  struct S; // expected-note {{member is declared here}}
  typeid(*static_cast<S *>(nullptr)); // expected-error {{implicit instantiation of undefined member 'S'}}
};

void test() {
  recursiveLambda(recursiveLambda, 10); // expected-note {{in instantiation of function template specialization}}
}
} // namespace gh176397

namespace gh63242 {
struct scoped {
  enum class scoped2 {
    RED,
    YELLOW,
    GREEN
  };
};

template <auto N>
struct scoped_struct {
  void f() {
    class scoped2 e = scoped::scoped2::RED; // expected-error {{implicit instantiation of undefined member 'scoped2'}} expected-note {{member is declared here}}
    (void)typeid(e);
  }
};

void test() {
  scoped_struct<scoped::scoped2::RED> s;
  s.f(); // expected-note {{in instantiation of member function 'gh63242::scoped_struct<gh63242::scoped::scoped2::RED>::f' requested here}}
}
} // namespace gh63242
