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

    initializer_list() : __begin_(nullptr), __size_(0) {}

    size_t    size()  const {return __size_;}
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

void argument_deduction() {
  static_assert(same_type<decltype(deduce({1, 2, 3})), int>::value, "bad deduction");
  static_assert(same_type<decltype(deduce({1.0, 2.0, 3.0})), double>::value, "bad deduction");

  deduce({1, 2.0}); // expected-error {{no matching function}}

  static_assert(same_type<decltype(deduce_ref({1, 2, 3})), int>::value, "bad deduction");
  static_assert(same_type<decltype(deduce_ref({1.0, 2.0, 3.0})), double>::value, "bad deduction");

  deduce_ref({1, 2.0}); // expected-error {{no matching function}}
}

void auto_deduction() {
  auto l = {1, 2, 3, 4};
  static_assert(same_type<decltype(l), std::initializer_list<int>>::value, "");
  auto bl = {1, 2.0}; // expected-error {{cannot deduce}}

  for (int i : {1, 2, 3, 4}) {}
}

void dangle() {
  new auto{1, 2, 3}; // expected-error {{cannot use list-initialization}}
  new std::initializer_list<int>{1, 2, 3}; // expected-warning {{at the end of the full-expression}}
}

struct haslist1 {
  std::initializer_list<int> il = {1, 2, 3}; // expected-warning{{at the end of the constructor}}
  std::initializer_list<int> jl{1, 2, 3}; // expected-warning{{at the end of the constructor}}
  haslist1();
};

haslist1::haslist1()
: il{1, 2, 3} // expected-warning{{at the end of the constructor}}
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
  auto x { { 0, 0 } }; // expected-error {{cannot deduce actual type for variable 'x' with type 'auto' from initializer list}}
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
  template<typename T> void g(std::initializer_list<T>);
  // expected-note@-1 {{candidate template ignored: couldn't resolve reference to overloaded function 'f'}}
  void h() { g({f}); }
  // expected-error@-1 {{no matching function for call to 'g'}}
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
