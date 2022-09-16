// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

// This must obviously come before the definition of std::initializer_list.
void missing_initializerlist() {
  auto l = {1, 2, 3, 4}; // expected-error {{std::initializer_list was not found}}
}

namespace std {
  typedef decltype(sizeof(int)) size_t;

  // libc++'s implementation
  template <class _E>
  class initializer_list
  {
    const _E* __begin_;
    size_t    __size_;

    initializer_list(const _E* __b, size_t __s)
      : __begin_(__b),
        __size_(__s)
    {}

  public:
    typedef _E        value_type;
    typedef const _E& reference;
    typedef const _E& const_reference;
    typedef size_t    size_type;

    typedef const _E* iterator;
    typedef const _E* const_iterator;

    constexpr initializer_list() : __begin_(nullptr), __size_(0) {}

    constexpr size_t    size()  const {return __size_;}
    const _E* begin() const {return __begin_;}
    const _E* end()   const {return __begin_ + __size_;}
  };
}

template <typename T, typename U>
struct same_type { static const bool value = false; };
template <typename T>
struct same_type<T, T> { static const bool value = true; };

struct one { char c[1]; };
struct two { char c[2]; };

struct A {
  int a, b;
};

struct B {
  B();
  B(int, int);
};

void simple_list() {
  std::initializer_list<int> il = { 1, 2, 3 };
  std::initializer_list<double> dl = { 1.0, 2.0, 3 };
  std::initializer_list<A> al = { {1, 2}, {2, 3}, {3, 4} };
  std::initializer_list<B> bl = { {1, 2}, {2, 3}, {} };
}

void function_call() {
  void f(std::initializer_list<int>);
  f({1, 2, 3});

  void g(std::initializer_list<B>);
  g({ {1, 2}, {2, 3}, {} });
}

struct C {
  C(int);
};

struct D {
  D();
  operator int();
  operator C();
};

void overloaded_call() {
    one overloaded(std::initializer_list<int>);
    two overloaded(std::initializer_list<B>);

    static_assert(sizeof(overloaded({1, 2, 3})) == sizeof(one), "bad overload");
    static_assert(sizeof(overloaded({ {1, 2}, {2, 3}, {} })) == sizeof(two), "bad overload");

    void ambiguous(std::initializer_list<A>); // expected-note {{candidate}}
    void ambiguous(std::initializer_list<B>); // expected-note {{candidate}}
    ambiguous({ {1, 2}, {2, 3}, {3, 4} }); // expected-error {{ambiguous}}

    one ov2(std::initializer_list<int>); // expected-note {{candidate}}
    two ov2(std::initializer_list<C>); // expected-note {{candidate}}
    // Worst sequence to int is identity, whereas to C it's user-defined.
    static_assert(sizeof(ov2({1, 2, 3})) == sizeof(one), "bad overload");
    // But here, user-defined is worst in both cases.
    ov2({1, 2, D()}); // expected-error {{ambiguous}}
}

template <typename T>
T deduce(std::initializer_list<T>); // expected-note {{conflicting types for parameter 'T' ('int' vs. 'double')}}
template <typename T>
T deduce_ref(const std::initializer_list<T>&); // expected-note {{conflicting types for parameter 'T' ('int' vs. 'double')}}

template<typename T, typename U> struct pair { pair(...); };
template<typename T> void deduce_pairs(std::initializer_list<pair<T, typename T::type>>);
// expected-note@-1 {{deduced type 'pair<[...], typename WithIntType::type>' of element of 1st parameter does not match adjusted type 'pair<[...], float>' of element of argument [with T = WithIntType]}}
struct WithIntType { typedef int type; };

template<typename ...T> void deduce_after_init_list_in_pack(void (*)(T...), T...); // expected-note {{<int, int> vs. <(no value), double>}}

void argument_deduction() {
  static_assert(same_type<decltype(deduce({1, 2, 3})), int>::value, "bad deduction");
  static_assert(same_type<decltype(deduce({1.0, 2.0, 3.0})), double>::value, "bad deduction");

  deduce({1, 2.0}); // expected-error {{no matching function}}

  static_assert(same_type<decltype(deduce_ref({1, 2, 3})), int>::value, "bad deduction");
  static_assert(same_type<decltype(deduce_ref({1.0, 2.0, 3.0})), double>::value, "bad deduction");

  deduce_ref({1, 2.0}); // expected-error {{no matching function}}

  pair<WithIntType, int> pi;
  pair<WithIntType, float> pf;
  deduce_pairs({pi, pi, pi}); // ok
  deduce_pairs({pi, pf, pi}); // expected-error {{no matching function}}

  deduce_after_init_list_in_pack((void(*)(int,int))0, {}, 0);
  deduce_after_init_list_in_pack((void(*)(int,int))0, {}, 0.0); // expected-error {{no matching function}}
}

void auto_deduction() {
  auto l = {1, 2, 3, 4};
  auto l2 {1, 2, 3, 4}; // expected-error {{initializer for variable 'l2' with type 'auto' contains multiple expressions}}
  auto l3 {1};
  static_assert(same_type<decltype(l), std::initializer_list<int>>::value, "");
  static_assert(same_type<decltype(l3), int>::value, "");
  auto bl = {1, 2.0}; // expected-error {{deduced conflicting types ('int' vs 'double') for initializer list element type}}

  void f1(int), f1(float), f2(int), f3(float);
  auto fil = {f1, f2};
  auto ffl = {f1, f3};
  auto fl = {f1, f2, f3}; // expected-error {{deduced conflicting types ('void (*)(int)' vs 'void (*)(float)') for initializer list element type}}

  for (int i : {1, 2, 3, 4}) {}
  for (int j : {1.0, 2.0, 3.0f, 4.0}) {} // expected-error {{deduced conflicting types ('double' vs 'float') for initializer list element type}}
}

void dangle() {
  new auto{1, 2, 3}; // expected-error {{new expression for type 'auto' contains multiple constructor arguments}}
  new std::initializer_list<int>{1, 2, 3}; // expected-warning {{at the end of the full-expression}}
}

struct haslist1 {
  std::initializer_list<int> il // expected-note {{declared here}}
    = {1, 2, 3}; // ok, unused
  std::initializer_list<int> jl{1, 2, 3}; // expected-note {{default member init}}
  haslist1();
};

haslist1::haslist1() // expected-error {{backing array for 'std::initializer_list' member 'jl' is a temporary object}}
: il{1, 2, 3} // expected-error {{backing array for 'std::initializer_list' member 'il' is a temporary object}}
{}

namespace PR12119 {
  // Deduction with nested initializer lists.
  template<typename T> void f(std::initializer_list<T>);
  template<typename T> void g(std::initializer_list<std::initializer_list<T>>);

  void foo() {
    f({0, {1}}); // expected-warning{{braces around scalar initializer}}
    g({{0, 1}, {2, 3}});
    std::initializer_list<int> il = {1, 2};
    g({il, {2, 3}});
  }
}

namespace Decay {
  template<typename T>
  void f(std::initializer_list<T>) {
    T x = 1; // expected-error{{cannot initialize a variable of type 'const char *' with an rvalue of type 'int'}}
  }

  void g() {
    f({"A", "BB", "CCC"}); // expected-note{{in instantiation of function template specialization 'Decay::f<const char *>' requested here}}

    auto x = { "A", "BB", "CCC" };
    std::initializer_list<const char *> *il = &x;

    for( auto s : {"A", "BB", "CCC", "DDD"}) { }
  }
}

namespace PR12436 {
  struct X {
    template<typename T>
    X(std::initializer_list<int>, T);
  };
  
  X x({}, 17);
}

namespace rdar11948732 {
  template<typename T> struct X {};

  struct XCtorInit {
    XCtorInit(std::initializer_list<X<int>>);
  };

  void f(X<int> &xi) {
    XCtorInit xc = { xi, xi };
  }
}

namespace PR14272 {
  auto x { { 0, 0 } }; // expected-error {{cannot deduce type for variable 'x' with type 'auto' from nested initializer list}}
}

namespace initlist_of_array {
  void f(std::initializer_list<int[2]>) {}
  void f(std::initializer_list<int[2][2]>) = delete;
  void h() {
    f({{1,2},{3,4}});
  }
}

namespace init_list_deduction_failure {
  void f();
  void f(int);
  // FIXME: It'd be nice to track that 'T' became a non-deduced context due to
  // overload resolution failure for 'f'.
  template<typename T> void g(std::initializer_list<T>);
  // expected-note@-1 {{candidate template ignored: couldn't infer template argument 'T'}}
  void h() {
    g({f}); // expected-error {{no matching function for call to 'g'}}
    g({f, h}); // ok
  }
}

namespace deleted_copy {
  struct X {
    X(int i) {}
    X(const X& x) = delete; // expected-note {{here}}
    void operator=(const X& x) = delete;
  };

  std::initializer_list<X> x{1}; // expected-error {{invokes deleted constructor}}
}

namespace RefVersusInitList {
  struct S {};
  void f(const S &) = delete;
  void f(std::initializer_list<S>);
  void g(S s) { f({S()}); }
}

namespace PR18013 {
  int f();
  std::initializer_list<long (*)()> x = {f}; // expected-error {{cannot initialize an array element of type 'long (*const)()' with an lvalue of type 'int ()': different return type ('long' vs 'int')}}
}

namespace DR1070 {
  struct S {
    S(std::initializer_list<int>);
  };
  S s[3] = { {1, 2, 3}, {4, 5} }; // ok
  S *p = new S[3] { {1, 2, 3}, {4, 5} }; // ok
}

namespace ListInitInstantiate {
  struct A {
    A(std::initializer_list<A>);
    A(std::initializer_list<int>);
  };
  struct B : A {
    B(int);
  };
  template<typename T> struct X {
    X();
    A a;
  };
  template<typename T> X<T>::X() : a{B{0}, B{1}} {}

  X<int> x;

  int f(const A&);
  template<typename T> void g() { int k = f({0}); }
  template void g<int>();
}

namespace TemporaryInitListSourceRange_PR22367 {
  struct A {
    constexpr A() {}
    A(std::initializer_list<int>); // expected-note {{here}}
  };
  constexpr int f(A) { return 0; }
  constexpr int k = f( // expected-error {{must be initialized by a constant expression}}
      // The point of this test is to check that the caret points to
      // 'std::initializer_list', not to '{0}'.
      std::initializer_list // expected-note {{constructor}}
      <int>
      {0}
      );
}

namespace ParameterPackNestedInitializerLists_PR23904c3 {
  template <typename ...T>
  void f(std::initializer_list<std::initializer_list<T>> ...tt); // expected-note 2{{conflicting}} expected-note {{incomplete pack}}

  void foo() {
    f({{0}}, {{'\0'}}); // ok, T = <int, char>
    f({{0}, {'\0'}}); // expected-error {{no match}}
    f({{0, '\0'}}); // expected-error {{no match}}

    f({{0}}, {{{}}}); // expected-error {{no match}}
    f({{0}}, {{{}, '\0'}}); // ok, T = <int, char>
    f({{0}, {{}}}); // ok, T = <int>
    f({{0, {}}}); // ok, T = <int>
  }
}

namespace update_rbrace_loc_crash {
  // We used to crash-on-invalid on this example when updating the right brace
  // location.
  template <typename T, T>
  struct A {};
  template <typename T, typename F, int... I>
  std::initializer_list<T> ExplodeImpl(F p1, A<int, I...>) {
    // expected-error@+1 {{reference to incomplete type 'const Incomplete' could not bind to an rvalue of type 'void'}}
    return {p1(I)...};
  }
  template <typename T, int N, typename F>
  void Explode(F p1) {
    // expected-note@+1 {{in instantiation of function template specialization}}
    ExplodeImpl<T>(p1, A<int, N>());
  }
  class Incomplete;
  struct ContainsIncomplete {
    const Incomplete &obstacle;
  };
  void f() {
    // expected-note@+1 {{in instantiation of function template specialization}}
    Explode<ContainsIncomplete, 4>([](int) {});
  }
}

namespace no_conversion_after_auto_list_deduction {
  // We used to deduce 'auto' == 'std::initializer_list<X>' here, and then
  // incorrectly accept the declaration of 'x'.
  struct X { using T = std::initializer_list<X> X::*; operator T(); };
  auto X::*x = { X() }; // expected-error {{from initializer list}}

  struct Y { using T = std::initializer_list<Y>(*)(); operator T(); };
  auto (*y)() = { Y() }; // expected-error {{from initializer list}}
}

namespace designated_init {
  constexpr auto a = {.a = 1, .b = 2}; // expected-error {{cannot deduce}}
  constexpr auto b = {[0] = 1, [4] = 2}; // expected-error {{cannot deduce}} expected-warning {{C99}}
  constexpr auto c = {1, [4] = 2}; // expected-error {{cannot deduce}} expected-warning 2{{C99}} expected-note {{here}}
  constexpr auto d = {1, [0] = 2}; // expected-error {{cannot deduce}} expected-warning 2{{C99}} expected-note {{here}}

  // If we ever start accepting the above, these assertions should pass.
  static_assert(c.size() == 5, "");
  static_assert(d.size() == 1, "");
}

namespace weird_initlist {
  struct weird {};
}
template<> struct std::initializer_list<weird_initlist::weird> { int a, b, c; };
namespace weird_initlist {
  // We don't check the struct layout in Sema.
  auto x = {weird{}, weird{}, weird{}, weird{}, weird{}};
  // ... but we do in constant evaluation.
  constexpr auto y = {weird{}, weird{}, weird{}, weird{}, weird{}}; // expected-error {{constant}} expected-note {{type 'const std::initializer_list<weird>' has unexpected layout}}
}

auto v = std::initializer_list<int>{1,2,3}; // expected-warning {{array backing local initializer list 'v' will be destroyed at the end of the full-expression}}

std::initializer_list<int> get(int cond) {
  if (cond == 0)
    return {};
  if (cond == 1)
    return {1, 2, 3}; // expected-warning {{returning address of local temporary object}}
  return std::initializer_list<int>{1, 2, 3}; // expected-warning {{returning address of local temporary object}}
}
