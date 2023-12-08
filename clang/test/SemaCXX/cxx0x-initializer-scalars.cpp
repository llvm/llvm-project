// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s -pedantic-errors

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

namespace integral {

  void initialization() {
    { const int a{}; static_assert(a == 0, ""); }
    { const int a = {}; static_assert(a == 0, ""); }
    { const int a{1}; static_assert(a == 1, ""); }
    { const int a = {1}; static_assert(a == 1, ""); }
    { const int a{1, 2}; } // expected-error {{excess elements}}
    { const int a = {1, 2}; } // expected-error {{excess elements}}
    // FIXME: Redundant warnings.
    { const short a{100000}; } // expected-error {{cannot be narrowed}} expected-note {{insert an explicit cast}} expected-warning {{changes value}}
    { const short a = {100000}; } // expected-error {{cannot be narrowed}} expected-note {{insert an explicit cast}} expected-warning {{changes value}}
    { if (const int a{1}) static_assert(a == 1, ""); }
    { if (const int a = {1}) static_assert(a == 1, ""); }
  }

  int direct_usage() {
    int ar[10];
    (void) ar[{1}]; // expected-error {{array subscript is not an integer}}

    return {1}; // expected-warning {{braces around scalar init}}
  }

  void inline_init() {
    auto v = int{1};
    (void) new int{1};
  }

  struct A {
    int i;
    A() : i{1} {}
  };

  void function_call() {
    void takes_int(int);
    takes_int({1}); // expected-warning {{braces around scalar init}}
  }

  void overloaded_call() {
    one overloaded(int);
    two overloaded(double);

    static_assert(sizeof(overloaded({0})) == sizeof(one), "bad overload"); // expected-warning {{braces around scalar init}}
    static_assert(sizeof(overloaded({0.0})) == sizeof(two), "bad overload"); // expected-warning {{braces around scalar init}}

    void ambiguous(int, double); // expected-note {{candidate}}
    void ambiguous(double, int); // expected-note {{candidate}}
    ambiguous({0}, {0}); // expected-error {{ambiguous}}

    void emptylist(int);
    void emptylist(int, int, int);
    emptylist({});
    emptylist({}, {}, {});
  }

  void edge_cases() {
    int a({0}); // expected-error {{cannot initialize non-class type 'int' with a parenthesized initializer list}}
    (void) int({0}); // expected-error {{cannot initialize non-class type 'int' with a parenthesized initializer list}}
    new int({0});  // expected-error {{cannot initialize non-class type 'int' with a parenthesized initializer list}}

    int *b({0});  // expected-error {{cannot initialize non-class type 'int *' with a parenthesized initializer list}}
    typedef int *intptr;
    int *c = intptr({0});  // expected-error {{cannot initialize non-class type 'intptr' (aka 'int *') with a parenthesized initializer list}}
  }

  template<typename T> void dependent_edge_cases() {
    T a({0});
    (void) T({0});
    new T({0});

    T *b({0});
    typedef T *tptr;
    T *c = tptr({0});
  }

  void default_argument(int i = {}) {
  }
  struct DefaultArgument {
    void default_argument(int i = {}) {
    }
  };
}

namespace PR12118 {
  void test() {
    one f(std::initializer_list<int>); 
    two f(int); 

    // to initializer_list is preferred
    static_assert(sizeof(f({0})) == sizeof(one), "bad overload");
  }
}

namespace excess_braces_sfinae {
  using valid = int&;
  using invalid = float&;

  template<typename T> valid braces1(decltype(T{0})*);
  template<typename T> invalid braces1(...);

  template<typename T> valid braces2(decltype(T{{0}})*);
  template<typename T> invalid braces2(...);

  template<typename T> valid braces3(decltype(T{{{0}}})*);
  template<typename T> invalid braces3(...);

  valid a = braces1<int>(0);
  invalid b = braces2<int>(0);
  invalid c = braces3<int>(0);

  struct X { int n; };
  valid d = braces1<X>(0);
  valid e = braces2<X>(0);
  invalid f = braces3<X>(0);
}
