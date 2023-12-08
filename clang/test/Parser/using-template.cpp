// RUN: %clang_cc1 %s -verify

namespace N1 {
template <typename... Ts>
struct Foo {
  template <typename T>
  struct Bar {
    static constexpr bool is_present = false;
  };
};

template <typename T, typename... Ts>
struct Foo<T, Ts...> : public Foo<Ts...> {
  using template Foo<Ts...>::Bar;
  // expected-error@-1 {{'template' keyword not permitted after 'using' keyword}}
};
}

namespace N2 {
namespace foo {
  using I = int;
}
using template namespace foo;
// expected-error@-1 {{'template' keyword not permitted after 'using' keyword}}
using template template namespace foo;
// expected-error@-1 2{{'template' keyword not permitted after 'using' keyword}}
I i;
}

namespace N3 {
namespace foo {
  using I = int;
}
using template foo::I;
// expected-error@-1 {{'template' keyword not permitted after 'using' keyword}}
I i;
}

namespace N4 {
template <typename T>
class A {};

template <typename T>
using B = A<T>;
B<int> b;

using template <typename T> C = A<T>;
// expected-error@-1 {{'template' keyword not permitted after 'using' keyword}}
// expected-error@-2 {{expected unqualified-id}}
C<int> c;
// expected-error@-1 {{no template named 'C'}}
}
