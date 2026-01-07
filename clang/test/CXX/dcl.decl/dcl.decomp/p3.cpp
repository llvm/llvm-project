// RUN: %clang_cc1 -std=c++1z -verify %s

using size_t = decltype(sizeof(0));

struct A { int x, y; };
struct B { int x, y; };

void no_tuple_size_1() { auto [x, y] = A(); } // ok, bind elementwise

namespace std { template<typename T> struct tuple_size; }
void no_tuple_size_2() { auto [x, y] = A(); } // ok, bind elementwise

struct Bad1 { int a, b; };
template<> struct std::tuple_size<Bad1> {};
void no_tuple_size_3() { auto [x, y] = Bad1(); } // ok, omitting value is valid after DR2386

struct Bad2 {};
template<> struct std::tuple_size<Bad2> { const int value = 5; };
void no_tuple_size_4() { auto [x, y] = Bad2(); } // expected-error {{cannot bind this type; 'std::tuple_size<Bad2>::value' is not a valid integral constant expression}}

template<> struct std::tuple_size<A> { static const int value = 3; };
template<> struct std::tuple_size<B> { enum { value = 3 }; };

void no_get_1() {
  {
    auto [a0, a1] = A(); // expected-error {{binds to 3 elements}}
    auto [b0, b1] = B(); // expected-error {{binds to 3 elements}}
  }
  auto [a0, a1, a2] = A(); // expected-error {{undeclared identifier 'get'}} expected-note {{in implicit initialization of binding declaration 'a0'}}
}

int get(A);

void no_get_2() {
  // FIXME: This diagnostic is not great.
  auto [a0, a1, a2] = A(); // expected-error {{undeclared identifier 'get'}} expected-note {{in implicit initialization of binding declaration 'a0'}}
}

template<int> float &get(A); // expected-note 2 {{no known conversion}}

void no_tuple_element_1() {
  auto [a0, a1, a2] = A(); // expected-error-re {{'std::tuple_element<0U{{L*}}, A>::type' does not name a type}} expected-note {{in implicit}}
}

namespace std { template<size_t, typename> struct tuple_element; } // expected-note 2{{here}}

void no_tuple_element_2() {
  auto [a0, a1, a2] = A(); // expected-error {{implicit instantiation of undefined template 'std::tuple_element<0, A>'}} expected-note {{in implicit}}
}

template<> struct std::tuple_element<0, A> { typedef float type; };

void no_tuple_element_3() {
  auto [a0, a1, a2] = A(); // expected-error {{implicit instantiation of undefined template 'std::tuple_element<1, A>'}} expected-note {{in implicit}}
}

template<> struct std::tuple_element<1, A> { typedef float &type; };
template<> struct std::tuple_element<2, A> { typedef const float &type; };

template<int N> auto get(B) -> int (&)[N + 1]; // expected-note 2 {{no known conversion}}
template<int N> struct std::tuple_element<N, B> { typedef int type[N +1 ]; };

template<typename T> struct std::tuple_size<const T> : std::tuple_size<T> {};
template<size_t N, typename T> struct std::tuple_element<N, const T> {
  typedef const typename std::tuple_element<N, T>::type type;
};

void referenced_type() {
  auto [a0, a1, a2] = A();
  auto [b0, b1, b2] = B();

  A a;
  B b;
  auto &[ar0, ar1, ar2] = a;
  auto &[br0, br1, br2] = b;

  auto &&[arr0, arr1, arr2] = A();
  auto &&[brr0, brr1, brr2] = B();

  const auto &[acr0, acr1, acr2] = A();
  const auto &[bcr0, bcr1, bcr2] = B();


  using Float = float;
  using Float = decltype(a0);
  using Float = decltype(ar0);
  using Float = decltype(arr0);

  using ConstFloat = const float;
  using ConstFloat = decltype(acr0);

  using FloatRef = float&;
  using FloatRef = decltype(a1);
  using FloatRef = decltype(ar1);
  using FloatRef = decltype(arr1);
  using FloatRef = decltype(acr1);

  using ConstFloatRef = const float&;
  using ConstFloatRef = decltype(a2);
  using ConstFloatRef = decltype(ar2);
  using ConstFloatRef = decltype(arr2);
  using ConstFloatRef = decltype(acr2);


  using Int1 = int[1];
  using Int1 = decltype(b0);
  using Int1 = decltype(br0);
  using Int1 = decltype(brr0);

  using ConstInt1 = const int[1];
  using ConstInt1 = decltype(bcr0);

  using Int2 = int[2];
  using Int2 = decltype(b1);
  using Int2 = decltype(br1);
  using Int2 = decltype(brr1);

  using ConstInt2 = const int[2];
  using ConstInt2 = decltype(bcr1);

  using Int3 = int[3];
  using Int3 = decltype(b2);
  using Int3 = decltype(br2);
  using Int3 = decltype(brr2);

  using ConstInt3 = const int[3];
  using ConstInt3 = decltype(bcr2);
}

struct C { template<int> int get() const; };
template<> struct std::tuple_size<C> { static const int value = 1; };
template<> struct std::tuple_element<0, C> { typedef int type; };

int member_get() {
  auto [c] = C();
  using T = int;
  using T = decltype(c);
  return c;
}

constexpr C c = C();
template<const C *p> void dependent_binding_PR40674() {
  const auto &[c] = *p;
  (void)c;
}

struct D {
  // FIXME: Emit a note here explaining why this was ignored.
  template<int> struct get {};
};
template<> struct std::tuple_size<D> { static const int value = 1; };
template<> struct std::tuple_element<0, D> { typedef D::get<0> type; };
void member_get_class_template() {
  auto [d] = D(); // expected-error {{no matching function for call to 'get'}} expected-note {{in implicit init}}
}

struct E {
  // FIXME: Emit a note here explaining why this was ignored.
  int get();
};
template<> struct std::tuple_size<E> { static const int value = 1; };
template<> struct std::tuple_element<0, E> { typedef int type; };
void member_get_non_template() {
  // FIXME: This diagnostic is not very good.
  auto [e] = E(); // expected-error {{no matching function for call to 'get'}} expected-note {{in implicit init}}
}

namespace ADL {
  struct X {};
};
template<int> int get(ADL::X);
template<> struct std::tuple_size<ADL::X> { static const int value = 1; };
template<> struct std::tuple_element<0, ADL::X> { typedef int type; };
void adl_only_bad() {
  auto [x] = ADL::X(); // expected-error {{undeclared identifier 'get'}} expected-note {{in implicit init}}
}

template<typename ElemType, typename GetTypeLV, typename GetTypeRV>
struct wrap {
  template<size_t> GetTypeLV get() &;
  template<size_t> GetTypeRV get() &&;
};
template<typename ET, typename GTL, typename GTR>
struct std::tuple_size<wrap<ET, GTL, GTR>> {
  static const int value = 1;
};
template<typename ET, typename GTL, typename GTR>
struct std::tuple_element<0, wrap<ET, GTL, GTR>> {
  using type = ET;
};

template<typename T> T &lvalue();

void test_value_category() {
  // If the declared variable is an lvalue reference, the operand to get is an
  // lvalue. Otherwise it's an xvalue.
  { auto [a] = wrap<int, void, int>(); }
  { auto &[a] = lvalue<wrap<int, int, void>>(); }
  { auto &&[a] = wrap<int, void, int>(); }
  // If the initializer (call to get) is an lvalue, the binding is an lvalue
  // reference to the element type. Otherwise it's an rvalue reference to the
  // element type.
  { auto [a] = wrap<int, void, int&>(); }
  { auto [a] = wrap<int&, void, int&>(); }
  { auto [a] = wrap<int&&, void, int&>(); } // ok, reference collapse to int&

  { auto [a] = wrap<int, void, int&&>(); }
  { auto [a] = wrap<int&, void, int&&>(); } // expected-error {{non-const lvalue reference to type 'int' cannot bind}} expected-note {{in implicit}}
  { auto [a] = wrap<const int&, void, int&&>(); }
  { auto [a] = wrap<int&&, void, int&&>(); }

  { auto [a] = wrap<int, void, float&>(); } // expected-error {{cannot bind}} expected-note {{implicit}}
  { auto [a] = wrap<const int, void, float&>(); } // ok, const int &a can bind to float
  { auto [a] = wrap<int, void, float>(); } // ok, int &&a can bind to float
}

namespace constant {
  struct Q {};
  template<int N> constexpr int get(Q &&) { return N * N; }
}
template<> struct std::tuple_size<constant::Q> { static const int value = 3; };
template<int N> struct std::tuple_element<N, constant::Q> { typedef int type; };
namespace constant {
  Q q;
  // This creates and lifetime-extends a temporary to hold the result of each get() call.
  auto [a, b, c] = q;    // expected-note {{temporary}}
  static_assert(a == 0); // expected-error {{constant expression}} expected-note {{temporary}}

  constexpr bool f() {
    auto [a, b, c] = q;
    return a == 0 && b == 1 && c == 4;
  }
  static_assert(f());

  constexpr int g() {
    int *p = nullptr;
    {
      auto [a, b, c] = q;
      p = &c;
    }
    return *p; // expected-note {{read of object outside its lifetime}}
  }
  static_assert(g() == 4); // expected-error {{constant}} expected-note {{in call to 'g()'}}
}

// P0961R1
struct InvalidMemberGet {
  int get();
  template <class T> int get();
  struct get {};
};
template <> struct std::tuple_size<InvalidMemberGet> { static constexpr size_t value = 1; };
template <> struct std::tuple_element<0, InvalidMemberGet> { typedef float type; };
template <size_t> float get(InvalidMemberGet) { return 0; }
int f() {
  InvalidMemberGet img;
  auto [x] = img;
  typedef decltype(x) same_as_float;
  typedef float same_as_float;
}

struct ValidMemberGet {
  int get();
  template <class T> int get() { return 0; }
  template <size_t N> float get() { return 0; }
};
template <> struct std::tuple_size<ValidMemberGet> { static constexpr size_t value = 1; };
template <> struct std::tuple_element<0, ValidMemberGet> { typedef float type; };
// Don't use this one; we should use the member get.
template <size_t N> int get(ValidMemberGet) { static_assert(N && false, ""); }
int f2() {
  ValidMemberGet img;
  auto [x] = img;
  typedef decltype(x) same_as_float;
  typedef float same_as_float;
}

struct Base1 {
  int get(); // expected-note{{member found by ambiguous name lookup}}
};
struct Base2 {
  template<int> int get(); // expected-note{{member found by ambiguous name lookup}}
};
struct Derived : Base1, Base2 {};

template <> struct std::tuple_size<Derived> { static constexpr size_t value = 1; };
template <> struct std::tuple_element<0, Derived> { typedef int type; };

auto [x] = Derived(); // expected-error{{member 'get' found in multiple base classes of different types}}

struct Base {
  template<int> int get();
};
struct UsingGet : Base {
  using Base::get;
};

template <> struct std::tuple_size<UsingGet> {
  static constexpr size_t value = 1;
};
template <> struct std::tuple_element<0, UsingGet> { typedef int type; };

auto [y] = UsingGet();
