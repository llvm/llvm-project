// RUN: %clang_cc1 -std=c++20 -fsyntax-only %s -verify

#define UIE __attribute__((using_if_exists))

namespace test_basic {
namespace NS {}

using NS::x UIE; // expected-note{{using declaration annotated with 'using_if_exists' here}}
x usex();        // expected-error{{reference to unresolved using declaration}}

using NotNS::x UIE; // expected-error{{use of undeclared identifier 'NotNS'}}

using NS::NotNS::x UIE; // expected-error{{no member named 'NotNS' in namespace 'test_basic::NS'}}
} // namespace test_basic

namespace test_redecl {
namespace NS {}

using NS::x UIE;
using NS::x UIE;

namespace NS1 {}
namespace NS2 {}
namespace NS3 {
int A();     // expected-note{{target of using declaration}}
struct B {}; // expected-note{{target of using declaration}}
int C();     // expected-note{{conflicting declaration}}
struct D {}; // expected-note{{conflicting declaration}}
} // namespace NS3

using NS1::A UIE;
using NS2::A UIE; // expected-note{{using declaration annotated with 'using_if_exists' here}} expected-note{{conflicting declaration}}
using NS3::A UIE; // expected-error{{target of using declaration conflicts with declaration already in scope}}
int i = A();      // expected-error{{reference to unresolved using declaration}}

using NS1::B UIE;
using NS2::B UIE; // expected-note{{conflicting declaration}} expected-note{{using declaration annotated with 'using_if_exists' here}}
using NS3::B UIE; // expected-error{{target of using declaration conflicts with declaration already in scope}}
B myB;            // expected-error{{reference to unresolved using declaration}}

using NS3::C UIE;
using NS2::C UIE; // expected-error{{target of using declaration conflicts with declaration already in scope}} expected-note{{target of using declaration}}
int j = C();

using NS3::D UIE;
using NS2::D UIE; // expected-error{{target of using declaration conflicts with declaration already in scope}} expected-note{{target of using declaration}}
D myD;
} // namespace test_redecl

namespace test_dependent {
template <class B>
struct S : B {
  using B::mf UIE;          // expected-note 3 {{using declaration annotated with 'using_if_exists' here}}
  using typename B::mt UIE; // expected-note{{using declaration annotated with 'using_if_exists' here}}
};

struct BaseEmpty {
};
struct BaseNonEmpty {
  void mf();
  typedef int mt;
};

template <class Base>
struct UseCtor : Base {
  using Base::Base UIE; // expected-error{{'using_if_exists' attribute cannot be applied to an inheriting constructor}}
};
struct BaseCtor {};

void f() {
  S<BaseEmpty> empty;
  S<BaseNonEmpty> nonempty;
  empty.mf(); // expected-error {{reference to unresolved using declaration}}
  nonempty.mf();
  (&empty)->mf(); // expected-error {{reference to unresolved using declaration}}
  (&nonempty)->mf();

  S<BaseEmpty>::mt y; // expected-error {{reference to unresolved using declaration}}
  S<BaseNonEmpty>::mt z;

  S<BaseEmpty>::mf(); // expected-error {{reference to unresolved using declaration}}

  UseCtor<BaseCtor> usector;
}

template <class B>
struct Implicit : B {
  using B::mf UIE;          // expected-note {{using declaration annotated with 'using_if_exists' here}}
  using typename B::mt UIE; // expected-note 2 {{using declaration annotated with 'using_if_exists' here}}

  void use() {
    mf(); // expected-error {{reference to unresolved using declaration}}
    mt x; // expected-error {{reference to unresolved using declaration}}
  }

  mt alsoUse(); // expected-error {{reference to unresolved using declaration}}
};

void testImplicit() {
  Implicit<BaseNonEmpty> nonempty;
  Implicit<BaseEmpty> empty; // expected-note {{in instantiation}}
  nonempty.use();
  empty.use(); // expected-note {{in instantiation}}
}

template <class>
struct NonDep : BaseEmpty {
  using BaseEmpty::x UIE; // expected-note{{using declaration annotated with 'using_if_exists' here}}
  x y();                  // expected-error{{reference to unresolved using declaration}}
};
} // namespace test_dependent

namespace test_using_pack {
template <class... Ts>
struct S : Ts... {
  using typename Ts::x... UIE; // expected-error 2 {{target of using declaration conflicts with declaration already in scope}} expected-note{{conflicting declaration}} expected-note{{target of using declaration}}
};

struct E1 {};
struct E2 {};
S<E1, E2> a;

struct F1 {
  typedef int x; // expected-note 2 {{conflicting declaration}}
};
struct F2 {
  typedef int x; // expected-note 2 {{target of using declaration}}
};
S<F1, F2> b;

S<E1, F2> c; // expected-note{{in instantiation of template class}}
S<F1, E2> d; // expected-note{{in instantiation of template class}}

template <class... Ts>
struct S2 : Ts... {
  using typename Ts::x... UIE; // expected-error 2 {{target of using declaration conflicts with declaration already in scope}} expected-note 3 {{using declaration annotated with 'using_if_exists' here}} expected-note{{conflicting declaration}} expected-note{{target of using declaration}}

  x mem(); // expected-error 3 {{reference to unresolved using declaration}}
};

S2<E1, E2> e; // expected-note{{in instantiation of template class}}
S2<F1, F2> f;
S2<E1, F2> g; // expected-note{{in instantiation of template class}}
S2<F1, E2> h; // expected-note{{in instantiation of template class}}

template <class... Ts>
struct S3 : protected Ts... {
  using Ts::m... UIE; // expected-error{{target of using declaration conflicts with declaration already in scope}} expected-note{{target of using declaration}}
};
struct B1 {
  enum { m }; // expected-note{{conflicting declaration}}
};
struct B2 {};

S3<B1, B2> i; // expected-note{{in instantiation of template}}
S<B2, B1> j;

} // namespace test_using_pack

namespace test_nested {
namespace NS {}

using NS::x UIE; // expected-note {{using declaration annotated with 'using_if_exists' here}}

namespace NS2 {
using ::test_nested::x UIE;
}

NS2::x y; // expected-error {{reference to unresolved using declaration}}
} // namespace test_nested

namespace test_scope {
int x; // expected-note{{conflicting declaration}}
void f() {
  int x; // expected-note{{conflicting declaration}}
  {
    using ::x UIE; // expected-note {{using declaration annotated with 'using_if_exists' here}}
    (void)x;       // expected-error {{reference to unresolved using declaration}}
  }

  {
    using test_scope::x;
    using ::x UIE; // expected-error{{target of using declaration conflicts with declaration already in scope}} expected-note{{target of using declaration}}
    (void)x;
  }

  (void)x;

  using ::x UIE; // expected-error{{target of using declaration conflicts with declaration already in scope}} expected-note{{target of using declaration}}
  (void)x;
}
} // namespace test_scope

namespace test_appertains_to {
namespace NS {
typedef int x;
}

// FIXME: This diagnostics is wrong.
using alias UIE = NS::x; // expected-error {{'using_if_exists' attribute only applies to named declarations, types, and value declarations}}

template <class>
using template_alias UIE = NS::x; // expected-error {{'using_if_exists' attribute only applies to named declarations, types, and value declarations}}

void f() UIE; // expected-error {{'using_if_exists' attribute only applies to named declarations, types, and value declarations}}

using namespace NS UIE; // expected-error {{'using_if_exists' attribute only applies to named declarations, types, and value declarations}}
} // namespace test_appertains_to

typedef int *fake_FILE;
int fake_printf();

namespace std {
using ::fake_FILE UIE;
using ::fake_printf UIE;
using ::fake_fopen UIE;  // expected-note {{using declaration annotated with 'using_if_exists' here}}
using ::fake_size_t UIE; // expected-note {{using declaration annotated with 'using_if_exists' here}}
} // namespace std

int main() {
  std::fake_FILE file;
  file = std::fake_fopen(); // expected-error {{reference to unresolved using declaration}} expected-error{{incompatible integer to pointer}}
  std::fake_size_t size;    // expected-error {{reference to unresolved using declaration}}
  size = fake_printf();
  size = std::fake_printf();
}
