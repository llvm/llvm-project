// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify %s

namespace std {
class type_info;
}

namespace gh176397 {
auto recursiveLambda = [](auto, int) {
  struct S; // #S-fwd
  typeid(*static_cast<S *>(nullptr));
  // expected-error@-1 {{'typeid' of incomplete type 'S'}}
  // expected-note@#S-fwd {{forward declaration of 'S'}}
  // expected-note@#call {{in instantiation of function template specialization}}
};

void test() {
  recursiveLambda(recursiveLambda, 10); // #call
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
    class scoped2 e = scoped::scoped2::RED; // #scoped2-fwd
    // expected-error@-1 {{implicit instantiation of undefined member 'scoped2'}}
    // expected-note@#scoped2-fwd {{member is declared here}}
    // expected-note@#f-call {{in instantiation of member function 'gh63242::scoped_struct<gh63242::scoped::scoped2::RED>::f' requested here}}

    (void)typeid(e);
    // expected-error@-1 {{'typeid' of incomplete type 'class scoped2'}}
    // expected-note@#scoped2-fwd {{forward declaration of 'scoped2'}}
  }
};

void test() {
  scoped_struct<scoped::scoped2::RED> s;
  s.f(); // #f-call
}
} // namespace gh63242
