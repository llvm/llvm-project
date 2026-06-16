// RUN: %clang_cc1 -std=c++2c %s -triple x86_64-unknown-linux-gnu -verify=expected
// RUN: %clang_cc1 -std=c++2c %s -triple x86_64-unknown-linux-gnu -verify=expected -fexperimental-new-constant-interpreter

namespace std {
  using size_t = decltype(sizeof(0));
  template<typename> struct tuple_size;
  template<size_t, typename> struct tuple_element;
}

struct Y { int n = 0; };
struct X { X(); X(Y); X(const X&); ~X(); int k = 42;}; // #X-decl
struct Z { constexpr Z(): i (43){}; int i;}; // #Z-decl
struct Z2 { constexpr Z2(): i (0){}; int i; ~Z2();}; // #Z2-decl

struct Bit { constexpr Bit(): i(1), j(1){}; int i: 2; int j:2;};

struct A { int a : 13; bool b; };

constexpr auto [a0, a1] = A(42, true);
static_assert(a0 == 42);
static_assert(a1 == true);

struct B {};
template<> struct std::tuple_size<B> { enum { value = 2 }; };
template<> struct std::tuple_size<const B> { enum { value = 2 }; };
template<> struct std::tuple_element<0, const B> { using type = Y; };
template<> struct std::tuple_element<1, const B> { using type = const int&; };
template<int N>
constexpr auto get(B) {
  if constexpr (N == 0)
    return Y();
  else
    return 0.0;
}


constexpr auto [t1] = Y {42};
static_assert(t1 == 42);

constexpr int i[] = {1, 2};
constexpr auto [t2, t3] = i;
static_assert(t2 == 1);
static_assert(t3 == 2);

constexpr auto [t4] = X();
// expected-error@-1 {{constexpr variable cannot have non-literal type 'const X'}} \
//   expected-note@#X-decl {{'X' is not literal because it is not an aggregate and has no constexpr constructors other than copy or move constructors}}

constexpr auto [t5] = Z();
static_assert(t5 == 43);

constexpr auto [t6] = Z2();
//expected-error@-1 {{constexpr variable cannot have non-literal type 'const Z2'}}
// expected-note@#Z2-decl {{'Z2' is not literal because its destructor is not constexpr}}

constexpr auto [t7, t8] = Bit();
static_assert(t7 == 1);
static_assert(t8 == 1);

void test_tpl(auto) {
    constexpr auto [...p] = Bit();
    static_assert(((p == 1) && ...));
}

void test() {
    test_tpl(0);
}

constexpr auto [a, b] = B {};
static_assert(a.n == 0);

constinit auto [init1] = Y {42};
constinit auto [init2] = X {};  
// expected-error@-1 {{variable does not have a constant initializer}} \
//   expected-note@-1 {{required by 'constinit' specifier here}} \
//   expected-note@-1 {{non-constexpr constructor 'X' cannot be used in a constant expression}} \
//   expected-note@#X-decl {{declared here}}

constexpr auto [init3] = X {}; 
// expected-error@-1 {{constexpr variable cannot have non-literal type 'const X'}}
//   expected-note@#X-decl {{'X' is not literal because it is not an aggregate and has no constexpr constructors other than copy or move constructors}}

struct C {};
template<> struct std::tuple_size<C> { constexpr static auto value = 1; };
template<> struct std::tuple_size<const C> { constexpr static auto value = 1; };
template<> struct std::tuple_element<0, const C> { using type = int; };
template<> struct std::tuple_element<0, C> { using type = int; };

template<int N>
auto get(C) { // #non-constexpr-get
  return 0;
};

auto [c1] = C();
static_assert(c1 == 0);
// expected-error@-1 {{static assertion expression is not an integral constant expression}} \
//   expected-note@-1 {{read of non-const variable 'c1' is not allowed in a constant expression}}
//   expected-note@-4 {{declared here}}

constexpr auto [c2] = C();
// expected-error@-1 {{constexpr variable 'c2' must be initialized by a constant expression}} \
//   expected-note@-1 {{in implicit initialization of binding declaration 'c2'}} \
//   expected-note@-1 {{non-constexpr function 'get<0>' cannot be used in a constant expression}} \
//   expected-note@#non-constexpr-get {{declared here}}

constinit auto [c3] = C();
// expected-error@-1 {{variable does not have a constant initializer}} \
//   expected-note@-1 {{in implicit initialization of binding declaration 'c3'}} \
//   expected-note@-1 {{required by 'constinit' specifier here}} \
//   expected-note@-1 {{non-constexpr function 'get<0>' cannot be used in a constant expression}}
//   expected-note@#non-constexpr-get {{declared here}}

struct D {};
template<> struct std::tuple_size<D> { constexpr static auto value = 1; };
template<> struct std::tuple_size<const D> { constexpr static auto value = 1; };
template<> struct std::tuple_element<0, D> { using type = int; };
template<> struct std::tuple_element<0, const D> { using type = int; };

template<int N>
constexpr auto get(D) {
  return 0;
};

auto [d1] = D();
static_assert(d1 == 0);
// expected-error@-1 {{static assertion expression is not an integral constant expression}} \
//   expected-note@-1 {{read of non-const variable 'd1' is not allowed in a constant expression}}
//   expected-note@-4 {{declared here}}

constexpr auto [d2] = D();
static_assert(d2 == 0);

constinit auto [d3] = D();
static_assert(d3 == 0);
// expected-error@-1 {{static assertion expression is not an integral constant expression}} \
//   expected-note@-1 {{read of non-const variable 'd3' is not allowed in a constant expression}}
//   expected-note@-4 {{declared here}}

struct E { bool a: 1; };
template<> struct std::tuple_size<E> { constexpr static auto value = 1; };
template<> struct std::tuple_size<const E> { constexpr static auto value = 1; };
template<> struct std::tuple_element<0, E> { using type = bool; };
template<> struct std::tuple_element<0, const E> { using type = bool const; };

template<int N>
constexpr auto const& get(E const& obj) {
  return obj.a; // #E-get
};

constexpr auto [e1] = E(true);
// expected-error@#E-get {{returning reference to local temporary object}} \
//   expected-note@-1 {{in instantiation of function template specialization 'get<0>' requested here}} \
//   expected-note@-1 {{in implicit initialization of binding declaration 'e1'}} \
// expected-error@-1 {{constexpr variable 'e1' must be initialized by a constant expression}} \
//   expected-note@-1 {{in implicit initialization of binding declaration 'e1'}} \
//   expected-note@-1 {{reference to temporary is not a constant expression}} \
//   expected-note@#E-get {{temporary created here}}
