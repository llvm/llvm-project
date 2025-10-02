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
// expected-note@#X-decl {{'X' is not literal because it is not an aggregate and has no constexpr constructors other than copy or move constructors}}

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

// FIXME : support tuple
constexpr auto [a, b] = B{};
static_assert(a.n == 0);
// expected-error@-1 {{static assertion expression is not an integral constant expression}} \
// expected-note@-1 {{read of temporary is not allowed in a constant expression outside the expression that created the temporary}}\
// expected-note@-2 {{temporary created here}}

constinit auto [init1] = Y {42};
constinit auto [init2] = X {};  // expected-error {{variable does not have a constant initializer}} \
// expected-note {{required by 'constinit' specifier here}} \
// expected-note {{non-constexpr constructor 'X' cannot be used in a constant expression}} \
// expected-note@#X-decl {{declared here}}
