// RUN: %clang_cc1 -std=c++2d -verify -fsyntax-only %s

template <class T> struct A { using type = T; };
template <class T> struct B { using type = T *; };
template <class...> struct List {};

namespace not_a_pack {
template <template <class> class TT>
using U = TT...[0]<int>; // expected-error {{'TT' does not refer to the name of a parameter pack}}

using V = A...[0]<int>; // expected-error {{'A' does not refer to the name of a parameter pack}}

template <template <class> concept CC>
constexpr bool W = CC...[0]<int>; // expected-error {{'CC' does not refer to the name of a parameter pack}}

template <template <class> auto VV>
constexpr int X = VV...[0]<int>; // expected-error {{'VV' does not refer to the name of a parameter pack}}
}

namespace index {
template <template <class> class... TT>
struct S {
  using ok = TT...[sizeof...(TT) - 1]<int>;
};
static_assert(__is_same(S<A, B>::ok, B<int>));

template <template <class> class... TT>
using OutOfBounds = TT...[2]<int>; // expected-error {{invalid index 2 for pack 'TT' of size 2}}
using E1 = OutOfBounds<A, B>;      // expected-note {{in instantiation of template type alias 'OutOfBounds' requested here}}

template <template <class> class... TT>
using Negative = TT...[-1]<int>;
// expected-error@-1 {{pack index evaluates to -1, which cannot be narrowed to type '__size_t'}}
// expected-error@-2 {{expected ';' after alias declaration}}

template <template <class> class... TT>
using Narrowing = TT...[1.0]<int>;
// expected-error@-1 {{conversion from 'double' to '__size_t' (aka 'unsigned long') is not allowed in a converted constant expression}}
// expected-error@-2 {{expected ';' after alias declaration}}

template <template <class> class... TT>
using NonConstant = TT...[x]<int>;
// expected-error@-1 {{use of undeclared identifier 'x'}}
// expected-error@-2 {{expected ';' after alias declaration}}
}

namespace equivalence {
template <template <class> class... TT>
void same(TT...[0]<int>);
template <template <class> class... TT>
void same(TT...[0]<int>);

template <unsigned N, template <class> class... TT>
void dependent(TT...[N]<int>);
template <unsigned N, template <class> class... TT>
void dependent(TT...[N]<int>);

template <template <class> class... TT>
void different(TT...[0]<int>);
template <template <class> class... TT>
void different(TT...[1]<int>);

void call() {
  same<A, B>(A<int>{});
  dependent<1, A, B>(B<int>{});
  different<A, B>(A<int>{});
  different<A, B>(B<int>{});
}
}

namespace uses {
template <class T> struct WithNested {
  using type = T;
  static constexpr int value = 1;
};

template <template <class> class... TT>
struct Everywhere : TT...[0]<int> {
  using alias = TT...[0]<int>;
  template <class T> using tmpl_alias = TT...[0]<T>;
  using nested = typename TT...[0]<int>::type;
  static constexpr int v = TT...[0]<int>::value;
  TT...[0]<int> member;
  TT...[0]<int> fn(TT...[0]<int>);
};

static_assert(__is_base_of(WithNested<int>, Everywhere<WithNested>));
static_assert(__is_same(Everywhere<WithNested>::alias, WithNested<int>));
static_assert(__is_same(Everywhere<WithNested>::tmpl_alias<char>, WithNested<char>));
static_assert(__is_same(Everywhere<WithNested>::nested, int));
static_assert(Everywhere<WithNested>::v == 1);

template <template <class> class> struct Take {};
template <template <class> class... TT>
using Arg = Take<TT...[1]>;
static_assert(__is_same(Arg<A, B>, Take<B>));

template <class T> struct Deduce {
  Deduce(T);
};
template <template <class> class... TT>
auto ctad() {
  TT...[0] x{42};
  return x;
}
static_assert(__is_same(decltype(ctad<Deduce>()), Deduce<int>));
}

namespace deduction_guides {
template <class T> struct C { C(T); };

template <template <class> class... TT>
TT...[0](int) -> TT...[0]<int>;
// expected-error@-1 {{expected unqualified-id}}
// expected-error@-2 {{expected ')'}}
// expected-note@-3 {{to match this '('}}

template <template <class> class... TT>
C(int) -> TT...[0]<int>;
// expected-error@-1 {{deduced type 'TT...[0]<int>' of deduction guide is not written as a specialization of template 'C'}}
}

namespace pack_expansion {
template <template <class> class... TT>
struct S {
  using one = TT...[0]<int>;
  template <unsigned... Is>
  using many = List<TT...[Is]<int>...>;
};
static_assert(__is_same(S<A, B>::many<1, 0>, List<B<int>, A<int>>));

template <template <class> class... TT>
struct Bad {
  template <unsigned... Is>
  using type = List<TT...[Is]<int>>; // expected-error {{declaration type contains unexpanded parameter pack 'Is'}}
};
}

namespace partial_substitution {
template <class T> struct Outer {
  template <template <class> class... TT>
  using inner = TT...[0]<T>;
};
static_assert(__is_same(Outer<int>::inner<A, B>, A<int>));

template <template <class> class... TT>
struct Nested {
  template <class... Ts>
  using apply = List<TT...[0]<Ts>...>;
};
static_assert(__is_same(Nested<A>::apply<int, long>, List<A<int>, A<long>>));
}

namespace empty_pack {
template <template <class> class... TT>
using U = TT...[0]<int>; // expected-error {{invalid index 0 for pack 'TT' of size 0}}
using E = U<>;           // expected-note {{in instantiation of template type alias 'U' requested here}}
}

namespace kinds {
template <class T> constexpr int Var = 1;
template <class T> constexpr int Var2 = 2;
template <class T> concept Always = true;
template <class T> concept Never = false;
template <class T, class U> concept Same = __is_same(T, U);

template <template <class> auto V1> struct TakeVar {};
template <template <class> concept C1> struct TakeConcept {};

template <template <class> auto... VV>
using AV = TakeVar<VV...[0]>;
static_assert(__is_same(AV<Var>, TakeVar<Var>));

template <template <class> concept... CC>
using AC = TakeConcept<CC...[0]>;
static_assert(__is_same(AC<Always>, TakeConcept<Always>));

template <template <class> auto... VV>
constexpr int use_var() {
  return VV...[1]<int>;
}
static_assert(use_var<Var, Var2>() == 2);
static_assert(use_var<Var2, Var>() == 1);

template <template <class> concept... CC>
constexpr bool use_concept() {
  return CC...[0]<int>;
}
static_assert(use_concept<Always, Never>());
static_assert(!use_concept<Never, Always>());

template <template <class> concept... CC>
struct Constrained {
  template <CC...[0] T>
  static constexpr int f() { return 1; }
};
static_assert(Constrained<Always, Never>::f<int>() == 1);

template <template <class, class> concept... CC>
struct ConstrainedWithArgs {
  template <CC...[0]<int> T>
  static constexpr int f() { return 2; }
};
static_assert(ConstrainedWithArgs<Same>::f<int>() == 2);

template <template <class> concept... CC, CC...[0] T>
constexpr int in_same_list(T) { return 3; }
static_assert(in_same_list<Always>(0) == 3);

template <template <class> concept... CC>
constexpr int with_auto(CC...[0] auto x) { return 4; }
static_assert(with_auto<Always>(0) == 4);

template <template <class> concept... CC>
constexpr int with_auto_var() {
  CC...[0] auto x = 5;
  return x;
}
static_assert(with_auto_var<Always>() == 5);

template <template <class> concept... CC>
constexpr auto with_auto_return() -> CC...[0] auto { return 6; }
static_assert(with_auto_return<Always>() == 6);

template <template <class> concept... CC>
constexpr int with_requires() requires CC...[0]<int> { return 7; }
static_assert(with_requires<Always>() == 7);

template <unsigned N, template <class> auto... VV>
constexpr int DependentVar = VV...[N]<int>;
static_assert(DependentVar<0, Var, Var2> == 1);
static_assert(DependentVar<1, Var, Var2> == 2);

template <unsigned N, template <class> concept... CC>
constexpr bool DependentConcept = CC...[N]<int>;
static_assert(DependentConcept<0, Always, Never>);
static_assert(!DependentConcept<1, Always, Never>);

template <template <class> concept... CC>
struct Expansion {
  template <unsigned... Is>
  static constexpr bool all() { return (CC...[Is]<int> && ...); }
};
static_assert(Expansion<Always, Always>::all<0, 1>());
static_assert(!Expansion<Always, Never>::all<0, 1>());

template <template <class> concept... CC>
constexpr bool GH218035 = ((CC...[0]<int> == CC<int>) && ...);
static_assert(GH218035<Always, Always>);
static_assert(!GH218035<Always, Never>);
static_assert(!GH218035<Never, Always>);

template <template <class> concept... CC>
struct GH218548 {
  template <class... Ts>
  static constexpr bool f() { return (CC...[sizeof(Ts) - 1]<int> && ...); }
};
static_assert(GH218548<Never, Always>::f<short>());
static_assert(!GH218548<Never, Always>::f<char>());
static_assert(!GH218548<Never, Always>::f<char, short>());
}

namespace kind_errors {
template <class T> constexpr int Var = 1;
template <class T> constexpr int Var2 = 2;
template <class T> concept Always = true;
template <class T> concept Big = sizeof(T) > 1;
// expected-note@-1 2{{because 'sizeof(char) > 1' (1 > 1) evaluated to false}}

template <template <class> concept... CC>
void constrained_auto(CC...[1] auto x);
// expected-note@-1 {{candidate template ignored: constraints not satisfied}}
// expected-note@-2 {{because 'char' does not satisfy 'Big'}}

void use_constrained_auto() {
  constrained_auto<Always, Big>('c'); // expected-error {{no matching function for call to 'constrained_auto'}}
  constrained_auto<Big, Always>('c');
}

template <template <class> concept... CC>
struct Constrained {
  template <CC...[0] T> // expected-note {{because 'char' does not satisfy 'Big'}}
  static void f();      // expected-note {{candidate template ignored: constraints not satisfied [with T = char]}}
};
void use_constrained() {
  Constrained<Big>::f<char>(); // expected-error {{no matching function for call to 'f'}}
  Constrained<Big>::f<int>();
}

template <template <class> auto... VV>
constexpr int VarOutOfBounds = VV...[2]<int>; // expected-error {{invalid index 2 for pack 'VV' of size 2}}
constexpr int E1 = VarOutOfBounds<Var, Var2>; // expected-note {{in instantiation of variable template specialization 'kind_errors::VarOutOfBounds<kind_errors::Var, kind_errors::Var2>' requested here}}

template <template <class> concept... CC>
constexpr bool ConceptOutOfBounds = CC...[1]<int>; // expected-error {{invalid index 1 for pack 'CC' of size 1}}
constexpr bool E2 = ConceptOutOfBounds<Always>;    // expected-note {{in instantiation of variable template specialization 'kind_errors::ConceptOutOfBounds<kind_errors::Always>' requested here}}

template <template <class> concept... CC>
struct ConstraintOutOfBounds {
  // expected-note@+1 {{because substituted constraint expression is ill-formed: invalid index 1 for pack 'CC' of size 1}}
  template <CC...[1] T>
  static void f(); // expected-note {{candidate template ignored: constraints not satisfied [with T = int]}}
};
void use_out_of_bounds() {
  ConstraintOutOfBounds<Always>::f<int>(); // expected-error {{no matching function for call to 'f'}}
}

template <auto> concept NonType = true;
template <template <auto> concept... CC>
struct NotATypeConcept {
  template <CC...[0] T> // expected-error {{concept named in type constraint is not a type concept}}
  static void f();
};
}

namespace sfinae {
template <class T> struct HasType {
  using type = T;
};
template <class T> struct NoType {};

template <template <class> class... TT>
constexpr int f(typename TT...[0]<int>::type *) { return 1; }
template <template <class> class... TT>
constexpr int f(...) { return 2; }

static_assert(f<HasType>(nullptr) == 1);
static_assert(f<NoType>(nullptr) == 2);
}

namespace equivalence {
template <class T> concept Always = true;

template <template <class> concept... CC>
void same(CC...[0] auto x) {} // expected-note {{previous definition is here}}
template <template <class> concept... CC>
void same(CC...[0] auto x) {} // expected-error {{redefinition of 'same'}}

template <template <class> concept... CC>
void different(CC...[0] auto x) {}
template <template <class> concept... CC>
void different(CC...[1] auto x) {}

template <template <class> concept... CC>
constexpr bool same_id = CC...[0]<int>; // expected-note {{previous definition is here}}
template <template <class> concept... CC>
constexpr bool same_id = CC...[0]<int>; // expected-error {{redefinition of 'same_id'}}

template <template <class> concept... CC>
constexpr bool different_id_a = CC...[0]<int>;
template <template <class> concept... CC>
constexpr bool different_id_b = CC...[1]<int>;

template <class T> concept Never = false;

template <template <class> concept... CC, class T> requires CC...[0]<T>
constexpr int same_constraint(T) { return 1; } // expected-note {{previous definition is here}}
template <template <class> concept... CC, class T> requires CC...[0]<T>
constexpr int same_constraint(T) { return 2; } // expected-error {{redefinition of 'same_constraint'}}

template <template <class> concept... CC, class T> requires CC...[0]<T>
constexpr int different_constraint(T) { return 1; }
template <template <class> concept... CC, class T> requires CC...[1]<T>
constexpr int different_constraint(T) { return 2; }
static_assert(different_constraint<Always, Never>(0) == 1);
static_assert(different_constraint<Never, Always>(0) == 2);
}

namespace atomic_constraints {
template <class T> concept Always = true;

template <template <class> concept... CC>
struct S {
  template <class T> requires CC...[0]<T>
  // expected-note@-1 {{similar constraint expression here}}
  static constexpr int f() { return 1; } // expected-note {{candidate function [with T = int]}}
  template <class T> requires CC...[0]<T> && Always<T>
  // expected-note@-1 {{similar constraint expressions not considered equivalent; constraint expressions cannot be considered equivalent unless they originate from the same concept}}
  static constexpr int f() { return 2; } // expected-note {{candidate function [with T = int]}}
};
constexpr int a = S<Always>::f<int>(); // expected-error {{call to 'f' is ambiguous}}

template <template <class> concept... CC>
struct T {
  template <class T> requires CC...[0]<T>
  static constexpr int f() { return 1; } // expected-note {{candidate function [with T = int]}}
  template <class T> requires Always<T>
  static constexpr int f() { return 2; } // expected-note {{candidate function [with T = int]}}
};
constexpr int b = T<Always>::f<int>(); // expected-error {{call to 'f' is ambiguous}}

template <template <class> concept... CC>
struct U {
  template <class T> requires CC...[0]<T>
  static constexpr int f() { return 1; } // expected-note {{candidate function [with T = int]}}
  template <class T> requires CC...[1]<T>
  static constexpr int f() { return 2; } // expected-note {{candidate function [with T = int]}}
};
constexpr int c = U<Always, Always>::f<int>(); // expected-error {{call to 'f' is ambiguous}}
}
