// RUN: %clang_cc1 -std=c++0x -fsyntax-only -fexceptions -verify %s

struct one { char c[1]; };
struct two { char c[2]; };

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

namespace objects {

  struct X1 { X1(int); };
  struct X2 { explicit X2(int); }; // expected-note {{constructor declared here}}

  template <int N>
  struct A {
    A() { static_assert(N == 0, ""); }
    A(int, double) { static_assert(N == 1, ""); }
  };

  template <int N>
  struct F {
    F() { static_assert(N == 0, ""); }
    F(int, double) { static_assert(N == 1, ""); }
    F(std::initializer_list<int>) { static_assert(N == 3, ""); }
  };

  template <int N>
  struct D {
    D(std::initializer_list<int>) { static_assert(N == 0, ""); } // expected-note 1 {{candidate}}
    D(std::initializer_list<double>) { static_assert(N == 1, ""); } // expected-note 1 {{candidate}}
  };

  template <int N>
  struct E {
    E(int, int) { static_assert(N == 0, ""); }
    E(X1, int) { static_assert(N == 1, ""); }
  };

  void overload_resolution() {
    { A<0> a{}; }
    { A<0> a = {}; }
    { A<1> a{1, 1.0}; }
    { A<1> a = {1, 1.0}; }

    { F<0> f{}; }
    { F<0> f = {}; }
    // Narrowing conversions don't affect viability. The next two choose
    // the initializer_list constructor.
    { F<3> f{1, 1.0}; } // expected-error {{type 'double' cannot be narrowed to 'int' in initializer list}} expected-note {{silence}}
    { F<3> f = {1, 1.0}; } // expected-error {{type 'double' cannot be narrowed to 'int' in initializer list}} expected-note {{silence}}
    { F<3> f{1, 2, 3, 4, 5, 6, 7, 8}; }
    { F<3> f = {1, 2, 3, 4, 5, 6, 7, 8}; }
    { F<3> f{1, 2, 3, 4, 5, 6, 7, 8}; }
    { F<3> f{1, 2}; }

    { D<0> d{1, 2, 3}; }
    { D<1> d{1.0, 2.0, 3.0}; }
    { D<-1> d{1, 2.0}; } // expected-error {{ambiguous}}

    { E<0> e{1, 2}; }
  }

  void explicit_implicit() {
    { X1 x{0}; }
    { X1 x = {0}; }
    { X2 x{0}; }
    { X2 x = {0}; } // expected-error {{constructor is explicit}}
  }

  struct C {
    C();
    C(int, double);
    C(int, int);

    int operator[](C);
  };

  C function_call() {
    void takes_C(C);
    takes_C({1, 1.0});

    C c;
    c[{1, 1.0}];

    return {1, 1.0};
  }

  void inline_init() {
    (void) C{1, 1.0};
    (void) new C{1, 1.0};
    (void) A<1>{1, 1.0};
    (void) new A<1>{1, 1.0};
  }

  struct B { // expected-note 2 {{candidate constructor}}
    B(C, int, C); // expected-note {{candidate constructor not viable: cannot convert initializer list argument to 'objects::C'}}
  };

  void nested_init() {
    B b1{{1, 1.0}, 2, {3, 4}};
    B b2{{1, 1.0, 4}, 2, {3, 4}}; // expected-error {{no matching constructor for initialization of 'objects::B'}}
  }

  void overloaded_call() {
    one ov1(B); // expected-note {{not viable: cannot convert initializer list}}
    two ov1(C); // expected-note {{not viable: cannot convert initializer list}}

    static_assert(sizeof(ov1({})) == sizeof(two), "bad overload");
    static_assert(sizeof(ov1({1, 2})) == sizeof(two), "bad overload");
    static_assert(sizeof(ov1({{1, 1.0}, 2, {3, 4}})) == sizeof(one), "bad overload");

    ov1({1}); // expected-error {{no matching function}}

    one ov2(int);
    two ov2(F<3>);
    // expected-warning@+1 {{braces around scalar initializer}}
    static_assert(sizeof(ov2({1})) == sizeof(one), "bad overload"); // list -> int ranks as identity
    static_assert(sizeof(ov2({1, 2, 3})) == sizeof(two), "bad overload"); // list -> F only viable
  }

  struct G { // expected-note 6 {{not viable}}
    // This is not an initializer-list constructor.
    template<typename ...T>
    G(std::initializer_list<int>, T ...);  // expected-note 3 {{not viable}}
  };

  struct H { // expected-note 6 {{not viable}}
    explicit H(int, int); // expected-note 3 {{not viable}} expected-note {{declared here}}
    H(int, void*); // expected-note 3 {{not viable}}
  };

  void edge_cases() {
    // invalid (the first phase only considers init-list ctors)
    // (for the second phase, no constructor is viable)
    G g1{1, 2, 3}; // expected-error {{no matching constructor}}
    (void) new G{1, 2, 3}; // expected-error {{no matching constructor}}
    (void) G{1, 2, 3} // expected-error {{no matching constructor}}

    // valid (T deduced to <>).
    G g2({1, 2, 3});
    (void) new G({1, 2, 3});
    (void) G({1, 2, 3});

    // invalid
    H h1({1, 2}); // expected-error {{no matching constructor}}
    (void) new H({1, 2}); // expected-error {{no matching constructor}}
    // FIXME: Bad diagnostic, mentions void type instead of init list.
    (void) H({1, 2}); // expected-error {{no matching conversion}}

    // valid (by copy constructor).
    H h2({1, nullptr});
    (void) new H({1, nullptr});
    (void) H({1, nullptr});

    // valid
    H h3{1, 2};
    (void) new H{1, 2};
    (void) H{1, 2};
  }

  struct memberinit {
    H h1{1, nullptr};
    H h2 = {1, nullptr};
    H h3{1, 1};
    H h4 = {1, 1}; // expected-error {{constructor is explicit}}
  };
}

namespace PR12092 {

  struct S {
    S(const char*);
  };
  struct V {
    template<typename T> V(T, T);
    void f(std::initializer_list<S>);
    void f(const V &);
  };

  void g() {
    extern V s;
    s.f({"foo", "bar"});
  }

}

namespace PR12117 {
  struct A { A(int); }; 
  struct B { B(A); } b{{0}};   //FIXME: non-conformant. Temporary fix until standard resolution.
                                // expected- error {{call to constructor of 'struct B' is ambiguous}} \
                                // expected- note 2{{candidate is the implicit}} \
                                // expected- note {{candidate constructor}}
  struct C { C(int); } c{0};
}

namespace PR12167 {
  template<int N> struct string {};

  struct X {
    X(const char v);
    template<typename T> bool operator()(T) const;
  };

  template<int N, class Comparator> bool g(const string<N>& s, Comparator cmp) {
    return cmp(s);
  }
  template<int N> bool f(const string<N> &s) {
    return g(s, X{'x'});
  }

  bool s = f(string<1>());
}

namespace PR12257_PR12241 {
  struct command_pair
  {
    command_pair(int, int);
  };

  struct command_map
  {
    command_map(std::initializer_list<command_pair>);
  };

  struct generator_pair
  {
    generator_pair(const command_map);
  };

  // 5 levels: init list, gen_pair, command_map, init list, command_pair
  const std::initializer_list<generator_pair> x = {{{{{3, 4}}}}};

  // 4 levels: init list, gen_pair, command_map via init list, command_pair
  const std::initializer_list<generator_pair> y = {{{{1, 2}}}};
}

namespace PR12120 {
  struct A { explicit A(int); A(float); }; // expected-note {{declared here}}
  A a = { 0 }; // expected-error {{constructor is explicit}}

  struct B { explicit B(short); B(long); }; // expected-note 4{{candidate}}
  B b = { 0 }; // expected-error {{ambiguous}}

  struct C { explicit C(short); C(long); }; // expected-note 2{{candidate}}
  C c = {{ 0 }}; // expected-error {{ambiguous}}
}

namespace PR12498 {
  class ArrayRef; // expected-note{{forward declaration}}

  struct C {
    void foo(const ArrayRef&); // expected-note{{passing argument to parameter here}}
  };

  static void bar(C* c)
  {
    c->foo({ nullptr, 1 }); // expected-error{{initialization of incomplete type 'const PR12498::ArrayRef'}}
  }
}

namespace explicit_default {
  struct A {
    explicit A(); // expected-note{{here}}
  };
  A a {}; // ok
  // This is copy-list-initialization, and we choose an explicit constructor
  // (even though we do so via value-initialization), so the initialization is
  // ill-formed.
  A b = {}; // expected-error{{chosen constructor is explicit}}
}

namespace init_list_default {
  struct A {
    A(std::initializer_list<int>);
  };
  A a {}; // calls initializer list constructor

  struct B {
    B();
    B(std::initializer_list<int>) = delete;
  };
  B b {}; // calls default constructor
}

// PR13470, <rdar://problem/11974632>
namespace PR13470 {
  struct W {
    explicit W(int); // expected-note {{here}}
  };

  struct X {
    X(const X&) = delete; // expected-note 3 {{here}}
    X(int);
  };

  template<typename T, typename Fn> void call(Fn f) {
    f({1}); // expected-error {{constructor is explicit}}
    f(T{1}); // expected-error {{call to deleted constructor}}
  }

  void ref_w(const W &); // expected-note 2 {{not viable}}
  void call_ref_w() {
    ref_w({1}); // expected-error {{no matching function}}
    ref_w(W{1});
    call<W>(ref_w); // expected-note {{instantiation of}}
  }

  void ref_x(const X &);
  void call_ref_x() {
    ref_x({1});
    ref_x(X{1});
    call<X>(ref_x); // ok
  }

  void val_x(X); // expected-note 2 {{parameter}}
  void call_val_x() {
    val_x({1});
    val_x(X{1}); // expected-error {{call to deleted constructor}}
    call<X>(val_x); // expected-note {{instantiation of}}
  }

  template<typename T>
  struct Y {
    X x{1};
    void f() { X x{1}; }
    void h() {
      ref_w({1}); // expected-error {{no matching function}}
      ref_w(W{1});
      ref_x({1});
      ref_x(X{1});
      val_x({1});
      val_x(X{1}); // expected-error {{call to deleted constructor}}
    }
    Y() {}
    Y(int) : x{1} {}
  };

  Y<int> yi;
  Y<int> yi2(0);
  void g() {
    yi.f();
    yi.h(); // ok, all diagnostics produced in template definition
  }
}

namespace PR19729 {
  struct A {
    A(int);
    A(const A&) = delete;
  };
  struct B {
    void *operator new(std::size_t, A);
  };
  B *p = new ({123}) B;
}

namespace PR11410 {
  struct A {
    A() = delete; // expected-note 2{{deleted here}}
    A(int);
  };

  A a[3] = {
    {1}, {2}
  }; // expected-error {{call to deleted constructor}} \
        expected-note {{in implicit initialization of array element 2 with omitted initializer}}

  struct B {
    A a; // expected-note {{in implicit initialization of field 'a'}}
  } b = {
  }; // expected-error {{call to deleted constructor}}

  struct C {
    C(int = 0); // expected-note 2{{candidate}}
    C(float = 0); // expected-note 2{{candidate}}
  };
  C c[3] = {
    0, 1
  }; // expected-error {{ambiguous}} expected-note {{in implicit initialization of array element 2}}
  C c2[3] = {
    [0] = 1, [2] = 3
  }; // expected-error {{ambiguous}} expected-note {{in implicit initialization of array element 1}}
}
