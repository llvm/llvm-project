// RUN: %clang_cc1 -std=c++98 %s -verify=expected -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 %s -verify=expected,cxx11-17,since-cxx11, -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 %s -verify=expected,cxx14-17,cxx11-17,since-cxx11,since-cxx14 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++17 %s -verify=expected,cxx14-17,cxx11-17,since-cxx11,since-cxx14 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++20 %s -verify=expected,since-cxx11,since-cxx14,since-cxx20 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++23 %s -verify=expected,since-cxx11,since-cxx14,since-cxx20 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++2c %s -verify=expected,since-cxx11,since-cxx14,since-cxx20 -fexceptions -fcxx-exceptions -pedantic-errors

namespace cwg1413 { // cwg1413: 12
  template<int> struct Check {
    typedef int type;
  };
  template<typename T> struct A : T {
    static const int a = 1;
    static const int b;
    static void c();
    void d();

    void f() {
      Check<true ? 0 : A::unknown_spec>::type *var1;
      // expected-error@-1 {{use of undeclared identifier 'var1'}}

      // ok, variable declaration
      Check<true ? 0 : a>::type *var2; // #cwg1413-var2
      Check<true ? 0 : b>::type *var3;
      // expected-error@-1 {{use of undeclared identifier 'var3'}}
      //   expected-note@#cwg1413-var2 {{'var2' declared here}}
      Check<true ? 0 : ((void)c, 0)>::type *var4;
      // expected-error@-1 {{use of undeclared identifier 'var4'}}
      //   expected-note@#cwg1413-var2 {{'var2' declared here}}

      // value-dependent because of the implied type-dependent 'this->', not because of 'd'
      Check<true ? 0 : (d(), 0)>::type *var5;
      // expected-error@-1 {{use of undeclared identifier 'var5'}}
      //   expected-note@#cwg1413-var2 {{'var2' declared here}}

      // value-dependent because of the value-dependent '&' operator, not because of 'A::d'
      Check<true ? 0 : (&A::d(), 0)>::type *var5;
      // expected-error@-1 {{use of undeclared identifier 'var5'}}
      //   expected-note@#cwg1413-var2 {{'var2' declared here}}
    }
  };
}

namespace cwg1423 { // cwg1423: 11
#if __cplusplus >= 201103L
  bool b1 = nullptr;
  // since-cxx11-error@-1 {{cannot initialize a variable of type 'bool' with an rvalue of type 'std::nullptr_t'}}
  bool b2(nullptr);
  // since-cxx11-warning@-1 {{implicit conversion of nullptr constant to 'bool'}}
  bool b3 = {nullptr};
  // since-cxx11-error@-1 {{cannot initialize a variable of type 'bool' with an rvalue of type 'std::nullptr_t'}}
  bool b4{nullptr};
  // since-cxx11-warning@-1 {{implicit conversion of nullptr constant to 'bool'}}
#endif
}

// cwg1425: na abi

namespace cwg1432 { // cwg1432: 16
#if __cplusplus >= 201103L
  template<typename T> T declval();

  template <class... T>
  struct common_type;

  template <class T, class U>
  struct common_type<T, U> {
   typedef decltype(true ? declval<T>() : declval<U>()) type;
  };

  template <class T, class U, class... V>
  struct common_type<T, U, V...> {
   typedef typename common_type<typename common_type<T, U>::type, V...>::type type;
  };

  template struct common_type<int, double>;
#endif
}

namespace cwg1443 { // cwg1443: yes
struct A {
  int i;
  A() { void foo(int=i); }
  // expected-error@-1 {{default argument references 'this'}}
};
}

namespace cwg1458 { // cwg1458: 3.1
#if __cplusplus >= 201103L
struct A;

void f() {
  constexpr A* a = nullptr;
  constexpr int p = &*a;
  // expected-error@-1 {{cannot initialize a variable of type 'const int' with an rvalue of type 'A *'}}
  constexpr A *p2 = &*a;
}

struct A {
  int operator&();
};
#endif
} // namespace cwg1458

namespace cwg1460 { // cwg1460: 3.5
#if __cplusplus >= 201103L
  namespace DRExample {
    union A {
      union {};
      // expected-error@-1 {{declaration does not declare anything}}
      union {};
      // expected-error@-1 {{declaration does not declare anything}}
      constexpr A() {}
    };
    constexpr A a = A();

    union B {
      union {};
      // expected-error@-1 {{declaration does not declare anything}}
      union {};
      // expected-error@-1 {{declaration does not declare anything}}
      constexpr B() = default;
    };
    constexpr B b = B();

    union C {
      union {};
      // expected-error@-1 {{declaration does not declare anything}}
      union {};
      // expected-error@-1 {{declaration does not declare anything}}
    };
    constexpr C c = C();
#if __cplusplus >= 201403L
    constexpr void f() { C c; }
    static_assert((f(), true), "");
#endif
  }

  union A {};
  union B { int n; }; // #cwg1460-B
  union C { int n = 0; };
  struct D { union {}; };
  // expected-error@-1 {{declaration does not declare anything}}
  struct E { union { int n; }; }; // #cwg1460-E
  struct F { union { int n = 0; }; };

  struct X {
    friend constexpr A::A() noexcept;
    friend constexpr B::B() noexcept;
    // cxx11-17-error@-1 {{constexpr declaration of 'B' follows non-constexpr declaration}}
    //   cxx11-17-note@#cwg1460-B {{previous declaration is here}}
    friend constexpr C::C() noexcept;
    friend constexpr D::D() noexcept;
    friend constexpr E::E() noexcept;
    // cxx11-17-error@-1 {{constexpr declaration of 'E' follows non-constexpr declaration}}
    //   cxx11-17-note@#cwg1460-E {{previous declaration is here}}
    friend constexpr F::F() noexcept;
  };

  // These are OK, because value-initialization doesn't actually invoke the
  // constructor.
  constexpr A a = A();
  constexpr B b = B();
  constexpr C c = C();
  constexpr D d = D();
  constexpr E e = E();
  constexpr F f = F();

  namespace Defaulted {
    union A { constexpr A() = default; };
    union B { int n; constexpr B() = default; };
    // cxx11-17-error@-1 {{defaulted definition of default constructor cannot be marked constexpr}}
    union C { int n = 0; constexpr C() = default; };
    struct D { union {}; constexpr D() = default; };
    // expected-error@-1 {{declaration does not declare anything}}
    struct E { union { int n; }; constexpr E() = default; };
    // cxx11-17-error@-1 {{defaulted definition of default constructor cannot be marked constexpr}}
    struct F { union { int n = 0; }; constexpr F() = default; };

    struct G { union { int n = 0; }; union { int m; }; constexpr G() = default; };
    // cxx11-17-error@-1 {{defaulted definition of default constructor cannot be marked constexpr}}
    struct H {
      union {
        int n = 0;
      };
      union { // #cwg1460-H-union
        int m;
      };
      constexpr H() {}
      // cxx11-17-error@-1 {{constexpr constructor that does not initialize all members is a C++20 extension}}
      //   cxx11-17-note@#cwg1460-H-union {{member not initialized by constructor}}
      constexpr H(bool) : m(1) {}
      constexpr H(char) : n(1) {}
      // cxx11-17-error@-1 {{constexpr constructor that does not initialize all members is a C++20 extension}}
      //   cxx11-17-note@#cwg1460-H-union {{member not initialized by constructor}}
      constexpr H(double) : m(1), n(1) {}
    };
  }

#if __cplusplus >= 201403L
  template<typename T> constexpr bool check() {
    T t; // #cwg1460-t
    return true;
  }
  static_assert(check<A>(), "");
  static_assert(check<B>(), ""); // #cwg1460-check-B
  // cxx14-17-error@-1 {{static assertion expression is not an integral constant expression}}
  //   cxx14-17-note@#cwg1460-t {{non-constexpr constructor 'B' cannot be used in a constant expression}}
  //   cxx14-17-note@#cwg1460-check-B {{in call to 'check<cwg1460::B>()'}}
  //   cxx14-17-note@#cwg1460-B {{declared here}}
  static_assert(check<C>(), "");
  static_assert(check<D>(), "");
  static_assert(check<E>(), ""); // #cwg1460-check-E
  // cxx14-17-error@-1 {{static assertion expression is not an integral constant expression}}
  //   cxx14-17-note@#cwg1460-t {{non-constexpr constructor 'E' cannot be used in a constant expression}}
  //   cxx14-17-note@#cwg1460-check-E {{in call to 'check<cwg1460::E>()'}}
  //   cxx14-17-note@#cwg1460-E {{declared here}}
  static_assert(check<F>(), "");
#endif

  union G {
    int a = 0; // #cwg1460-G-a
    int b = 0;
    // expected-error@-1 {{initializing multiple members of union}}
    //   expected-note@#cwg1460-G-a {{previous initialization is here}}
  };
  union H {
    union {
      int a = 0; // #cwg1460-H-a
    };
    union {
      int b = 0;
      // expected-error@-1 {{initializing multiple members of union}}
      //   expected-note@#cwg1460-H-a {{previous initialization is here}}
    };
  };
  struct I {
    union {
      int a = 0; // #cwg1460-I-a
      int b = 0;
      // expected-error@-1 {{initializing multiple members of union}}
      //   expected-note@#cwg1460-I-a {{previous initialization is here}}
    };
  };
  struct J {
    union { int a = 0; };
    union { int b = 0; };
  };

  namespace Overriding {
    struct A {
      int a = 1, b, c = 3;
      constexpr A() : b(2) {}
    };
    static_assert(A().a == 1 && A().b == 2 && A().c == 3, "");

    union B {
      int a, b = 2, c;
      constexpr B() : a(1) {}
      constexpr B(char) : b(4) {}
      constexpr B(int) : c(3) {}
      constexpr B(const char*) {}
    };
    static_assert(B().a == 1, "");
    static_assert(B().b == 2, "");
    // expected-error@-1 {{static assertion expression is not an integral constant expression}}
    //   expected-note@-2 {{read of member 'b' of union with active member 'a' is not allowed in a constant expression}}
    static_assert(B('x').a == 0, "");
    // expected-error@-1 {{static assertion expression is not an integral constant expression}}
    //   expected-note@-2 {{read of member 'a' of union with active member 'b' is not allowed in a constant expression}}
    static_assert(B('x').b == 4, "");
    static_assert(B(123).b == 2, "");
    // expected-error@-1 {{static assertion expression is not an integral constant expression}}
    //   expected-note@-2 {{read of member 'b' of union with active member 'c' is not allowed in a constant expression}}
    static_assert(B(123).c == 3, "");
    static_assert(B("").a == 1, "");
    // expected-error@-1 {{static assertion expression is not an integral constant expression}}
    //   expected-note@-2 {{read of member 'a' of union with active member 'b' is not allowed in a constant expression}}
    static_assert(B("").b == 2, "");
    static_assert(B("").c == 3, "");
    // expected-error@-1 {{static assertion expression is not an integral constant expression}}
    //   expected-note@-2 {{read of member 'c' of union with active member 'b' is not allowed in a constant expression}}

    struct C {
      union { int a, b = 2, c; };
      union { int d, e = 5, f; };
      constexpr C() : a(1) {}
      constexpr C(char) : c(3) {}
      constexpr C(int) : d(4) {}
      constexpr C(float) : f(6) {}
      constexpr C(const char*) {}
    };

    static_assert(C().a == 1, "");
    static_assert(C().b == 2, "");
    // expected-error@-1 {{static assertion expression is not an integral constant expression}}
    //   expected-note@-2 {{read of member 'b' of union with active member 'a' is not allowed in a constant expression}}
    static_assert(C().d == 4, "");
    // expected-error@-1 {{static assertion expression is not an integral constant expression}}
    //   expected-note@-2 {{read of member 'd' of union with active member 'e' is not allowed in a constant expression}}
    static_assert(C().e == 5, "");

    static_assert(C('x').b == 2, "");
    // expected-error@-1 {{static assertion expression is not an integral constant expression}}
    //   expected-note@-2 {{read of member 'b' of union with active member 'c' is not allowed in a constant expression}}
    static_assert(C('x').c == 3, "");
    static_assert(C('x').d == 4, "");
    // expected-error@-1 {{static assertion expression is not an integral constant expression}}
    //   expected-note@-2 {{read of member 'd' of union with active member 'e' is not allowed in a constant expression}}
    static_assert(C('x').e == 5, "");

    static_assert(C(1).b == 2, "");
    static_assert(C(1).c == 3, "");
    // expected-error@-1 {{static assertion expression is not an integral constant expression}}
    //   expected-note@-2 {{read of member 'c' of union with active member 'b' is not allowed in a constant expression}}
    static_assert(C(1).d == 4, "");
    static_assert(C(1).e == 5, "");
    // expected-error@-1 {{static assertion expression is not an integral constant expression}}
    //   expected-note@-2 {{read of member 'e' of union with active member 'd' is not allowed in a constant expression}}

    static_assert(C(1.f).b == 2, "");
    static_assert(C(1.f).c == 3, "");
    // expected-error@-1 {{static assertion expression is not an integral constant expression}}
    //   expected-note@-2 {{read of member 'c' of union with active member 'b' is not allowed in a constant expression}}
    static_assert(C(1.f).e == 5, "");
    // expected-error@-1 {{static assertion expression is not an integral constant expression}}
    //   expected-note@-2 {{read of member 'e' of union with active member 'f' is not allowed in a constant expression}}
    static_assert(C(1.f).f == 6, "");

    static_assert(C("").a == 1, "");
    // expected-error@-1 {{static assertion expression is not an integral constant expression}}
    //   expected-note@-2 {{read of member 'a' of union with active member 'b' is not allowed in a constant expression}}
    static_assert(C("").b == 2, "");
    static_assert(C("").c == 3, "");
    // expected-error@-1 {{static assertion expression is not an integral constant expression}}
    //   expected-note@-2 {{read of member 'c' of union with active member 'b' is not allowed in a constant expression}}
    static_assert(C("").d == 4, "");
    // expected-error@-1 {{static assertion expression is not an integral constant expression}}
    //   expected-note@-2 {{read of member 'd' of union with active member 'e' is not allowed in a constant expression}}
    static_assert(C("").e == 5, "");
    static_assert(C("").f == 6, "");
    // expected-error@-1 {{static assertion expression is not an integral constant expression}}
    //   expected-note@-2 {{read of member 'f' of union with active member 'e' is not allowed in a constant expression}}

    struct D;
    extern const D d;
    struct D {
      int a;
      union {
        int b = const_cast<D&>(d).a = 1; // not evaluated
        int c;
      };
      constexpr D() : a(0), c(0) {}
    };
    constexpr D d {};
    static_assert(d.a == 0, "");
  }
#endif
}

#if __cplusplus >= 201103L
namespace std {
  typedef decltype(sizeof(int)) size_t;

  // libc++'s implementation
  template <class _E>
  class initializer_list
  {
    const _E* __begin_;
    size_t    __size_;

    initializer_list(const _E* __b, size_t __s)
    : __begin_(__b), __size_(__s) {}

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
} // std
#endif

namespace cwg1467 {  // cwg1467: 3.7 c++11
#if __cplusplus >= 201103L
  // Note that the change to [over.best.ics] was partially undone by CWG2076;
  // the resulting rule is tested with the tests for that change.

  // List-initialization of aggregate from same-type object

  namespace basic0 {
    struct S {
      int i = 42;
    };

    S a;
    S b(a);
    S c{a};

    struct SS : public S { } x;
    S y(x);
    S z{x};
  } // basic0

  namespace basic1 {
    struct S {
      int i{42};
    };

    S a;
    S b(a);
    S c{a};

    struct SS : public S { } x;
    S y(x);
    S z{x};
  } // basic1

  namespace basic2 {
    struct S {
      int i = {42};
    };

    S a;
    S b(a);
    S c{a};

    struct SS : public S { } x;
    S y(x);
    S z{x};
  } // basic2

  namespace dr_example {
    struct OK {
      OK() = default;
      OK(const OK&) = default;
      OK(int) { }
    };

    OK ok;
    OK ok2{ok};

    struct X {
      X() = default;
      X(const X&) = default;
    };

    X x;
    X x2{x};

    void f1(int); // #cwg1467-f1
    void f1(std::initializer_list<long>) = delete; // #cwg1467-f1-deleted
    void g1() { f1({42}); }
    // since-cxx11-error@-1 {{call to deleted function 'f1'}}
    //   since-cxx11-note@#cwg1467-f1 {{candidate function}}
    //   since-cxx11-note@#cwg1467-f1-deleted {{candidate function has been explicitly deleted}}

    template <class T, class U>
    struct Pair {
      Pair(T, U);
    };
    struct String {
      String(const char *);
    };

    void f2(Pair<const char *, const char *>); // #cwg1467-f2
    void f2(std::initializer_list<String>) = delete; // #cwg1467-f2-deleted
    void g2() { f2({"foo", "bar"}); }
    // since-cxx11-error@-1 {{call to deleted function 'f2'}}
    //   since-cxx11-note@#cwg1467-f2 {{candidate function}}
    //   since-cxx11-note@#cwg1467-f2-deleted {{candidate function has been explicitly deleted}}
  } // dr_example

  namespace nonaggregate {
    struct NonAggregate {
      NonAggregate() {}
    };

    struct WantsIt {
      WantsIt(NonAggregate);
    };

    void f(NonAggregate);
    void f(WantsIt);

    void test1() {
      NonAggregate n;
      f({n});
    }

    void test2() {
      NonAggregate x;
      NonAggregate y{x};
      NonAggregate z{{x}};
    }
  } // nonaggregate

  struct NestedInit { int a, b, c; };
  NestedInit ni[1] = {{NestedInit{1, 2, 3}}};

  namespace NestedInit2 {
    struct Pair { int a, b; };
    struct TwoPairs { TwoPairs(Pair, Pair); };
    struct Value { Value(Pair); Value(TwoPairs); };
    void f() { Value{{{1,2},{3,4}}}; }
  }
  namespace NonAmbiguous {
  // The original implementation made this case ambiguous due to the special
  // handling of one element initialization lists.
  void f(int(&&)[1]);
  void f(unsigned(&&)[1]);

  void g(unsigned i) {
    f({i});
  }
  } // namespace NonAmbiguous

  namespace StringLiterals {
  // When the array size is 4 the call will attempt to bind an lvalue to an
  // rvalue and fail. Therefore #2 will be called. (rsmith will bring this
  // issue to CWG)
  void f(const char(&&)[4]);              // #cwg1467-f-char-4
  void f(const char(&&)[5]) = delete;     // #cwg1467-f-char-5
  void f(const wchar_t(&&)[4]);           // #cwg1467-f-wchar-4
  void f(const wchar_t(&&)[5]) = delete;  // #cwg1467-f-wchar-5
#if __cplusplus >= 202002L
  void f2(const char8_t(&&)[4]);          // #cwg1467-f2-char8-4
  void f2(const char8_t(&&)[5]) = delete; // #cwg1467-f2-char8-5
#endif
  void f(const char16_t(&&)[4]);          // #cwg1467-f-char16-4
  void f(const char16_t(&&)[5]) = delete; // #cwg1467-f-char16-5
  void f(const char32_t(&&)[4]);          // #cwg1467-f-char32-4
  void f(const char32_t(&&)[5]) = delete; // #cwg1467-f-char32-5
  void g() {
    f({"abc"});
    // since-cxx11-error@-1 {{call to deleted function 'f'}}
    //   since-cxx11-note@#cwg1467-f-char-5 {{candidate function has been explicitly deleted}}
    //   since-cxx11-note@#cwg1467-f-char-4 {{candidate function not viable: expects an rvalue for 1st argument}}
    //   since-cxx11-note@#cwg1467-f-wchar-4 {{candidate function not viable: no known conversion from 'const char[4]' to 'const wchar_t' for 1st argument}}
    //   since-cxx11-note@#cwg1467-f-wchar-5 {{candidate function not viable: no known conversion from 'const char[4]' to 'const wchar_t' for 1st argument}}
    //   since-cxx11-note@#cwg1467-f-char16-4 {{candidate function not viable: no known conversion from 'const char[4]' to 'const char16_t' for 1st argument}}
    //   since-cxx11-note@#cwg1467-f-char16-5 {{candidate function not viable: no known conversion from 'const char[4]' to 'const char16_t' for 1st argument}}
    //   since-cxx11-note@#cwg1467-f-char32-4 {{candidate function not viable: no known conversion from 'const char[4]' to 'const char32_t' for 1st argument}}
    //   since-cxx11-note@#cwg1467-f-char32-5 {{candidate function not viable: no known conversion from 'const char[4]' to 'const char32_t' for 1st argument}}
    f({((("abc")))});
    // since-cxx11-error@-1 {{call to deleted function 'f'}}
    //   since-cxx11-note@#cwg1467-f-char-5 {{candidate function has been explicitly deleted}}
    //   since-cxx11-note@#cwg1467-f-char-4 {{candidate function not viable: expects an rvalue for 1st argument}}
    //   since-cxx11-note@#cwg1467-f-wchar-4 {{candidate function not viable: no known conversion from 'const char[4]' to 'const wchar_t' for 1st argument}}
    //   since-cxx11-note@#cwg1467-f-wchar-5 {{candidate function not viable: no known conversion from 'const char[4]' to 'const wchar_t' for 1st argument}}
    //   since-cxx11-note@#cwg1467-f-char16-4 {{candidate function not viable: no known conversion from 'const char[4]' to 'const char16_t' for 1st argument}}
    //   since-cxx11-note@#cwg1467-f-char16-5 {{candidate function not viable: no known conversion from 'const char[4]' to 'const char16_t' for 1st argument}}
    //   since-cxx11-note@#cwg1467-f-char32-4 {{candidate function not viable: no known conversion from 'const char[4]' to 'const char32_t' for 1st argument}}
    //   since-cxx11-note@#cwg1467-f-char32-5 {{candidate function not viable: no known conversion from 'const char[4]' to 'const char32_t' for 1st argument}}
    f({L"abc"});
    // since-cxx11-error@-1 {{call to deleted function 'f'}}
    //   since-cxx11-note@#cwg1467-f-wchar-5 {{candidate function has been explicitly deleted}}
    //   since-cxx11-note@#cwg1467-f-char-4 {{candidate function not viable: no known conversion from 'const wchar_t[4]' to 'const char' for 1st argument}}
    //   since-cxx11-note@#cwg1467-f-char-5 {{candidate function not viable: no known conversion from 'const wchar_t[4]' to 'const char' for 1st argument}}
    //   since-cxx11-note@#cwg1467-f-wchar-4 {{candidate function not viable: expects an rvalue for 1st argument}}
    //   since-cxx11-note@#cwg1467-f-char16-4 {{candidate function not viable: no known conversion from 'const wchar_t[4]' to 'const char16_t' for 1st argument}}
    //   since-cxx11-note@#cwg1467-f-char16-5 {{candidate function not viable: no known conversion from 'const wchar_t[4]' to 'const char16_t' for 1st argument}}
    //   since-cxx11-note@#cwg1467-f-char32-4 {{candidate function not viable: no known conversion from 'const wchar_t[4]' to 'const char32_t' for 1st argument}}
    //   since-cxx11-note@#cwg1467-f-char32-5 {{candidate function not viable: no known conversion from 'const wchar_t[4]' to 'const char32_t' for 1st argument}}
#if __cplusplus >= 202002L
    f2({u8"abc"});
    // since-cxx20-error@-1 {{call to deleted function 'f2'}}
    //   since-cxx20-note@#cwg1467-f2-char8-5 {{candidate function has been explicitly deleted}}
    //   since-cxx20-note@#cwg1467-f2-char8-4 {{candidate function not viable: expects an rvalue for 1st argument}}
#endif
    f({uR"(abc)"});
    // since-cxx11-error@-1 {{call to deleted function 'f'}}
    //   since-cxx11-note@#cwg1467-f-char16-5 {{candidate function has been explicitly deleted}}
    //   since-cxx11-note@#cwg1467-f-char-4 {{candidate function not viable: no known conversion from 'const char16_t[4]' to 'const char' for 1st argument}}
    //   since-cxx11-note@#cwg1467-f-char-5 {{candidate function not viable: no known conversion from 'const char16_t[4]' to 'const char' for 1st argument}}
    //   since-cxx11-note@#cwg1467-f-wchar-4 {{candidate function not viable: no known conversion from 'const char16_t[4]' to 'const wchar_t' for 1st argument}}
    //   since-cxx11-note@#cwg1467-f-wchar-5 {{candidate function not viable: no known conversion from 'const char16_t[4]' to 'const wchar_t' for 1st argument}}
    //   since-cxx11-note@#cwg1467-f-char16-4 {{candidate function not viable: expects an rvalue for 1st argument}}
    //   since-cxx11-note@#cwg1467-f-char32-4 {{candidate function not viable: no known conversion from 'const char16_t[4]' to 'const char32_t' for 1st argument}}
    //   since-cxx11-note@#cwg1467-f-char32-5 {{candidate function not viable: no known conversion from 'const char16_t[4]' to 'const char32_t' for 1st argument}}
    f({(UR"(abc)")});
    // since-cxx11-error@-1 {{call to deleted function 'f'}}
    //   since-cxx11-note@#cwg1467-f-char32-5 {{candidate function has been explicitly deleted}}
    //   since-cxx11-note@#cwg1467-f-char-4 {{candidate function not viable: no known conversion from 'const char32_t[4]' to 'const char' for 1st argument}}
    //   since-cxx11-note@#cwg1467-f-char-5 {{candidate function not viable: no known conversion from 'const char32_t[4]' to 'const char' for 1st argument}}
    //   since-cxx11-note@#cwg1467-f-wchar-4 {{candidate function not viable: no known conversion from 'const char32_t[4]' to 'const wchar_t' for 1st argument}}
    //   since-cxx11-note@#cwg1467-f-wchar-5 {{candidate function not viable: no known conversion from 'const char32_t[4]' to 'const wchar_t' for 1st argument}}
    //   since-cxx11-note@#cwg1467-f-char16-4 {{candidate function not viable: no known conversion from 'const char32_t[4]' to 'const char16_t' for 1st argument}}
    //   since-cxx11-note@#cwg1467-f-char16-5 {{candidate function not viable: no known conversion from 'const char32_t[4]' to 'const char16_t' for 1st argument}}
    //   since-cxx11-note@#cwg1467-f-char32-4 {{candidate function not viable: expects an rvalue for 1st argument}}
  }
  } // namespace StringLiterals
#endif
} // cwg1467

namespace cwg1477 { // cwg1477: 2.7
namespace N {
struct A {
  // Name "f" is not bound in N,
  // so single searches of 'f' in N won't find it,
  // but the targets scope of this declaration is N,
  // making it nominable in N.
  // (_N4988_.[dcl.meaning]/2.1, [basic.scope.scope]/7,
  //  [basic.lookup.general]/3)
  friend int f();
};
}
// Corresponds to the friend declaration,
// because it's nominable in N,
// and binds name 'f' in N.
// (_N4988_.[dcl.meaning]/3.4, [basic.scope.scope]/2.5)
int N::f() { return 0; }
// Name 'f' is bound in N,
// so the search performed by qualified lookup finds it.
int i = N::f();
} // namespace cwg1477

namespace cwg1479 { // cwg1479: 3.1
#if __cplusplus >= 201103L
  int operator"" _a(const char*, std::size_t = 0);
  // since-cxx11-error@-1 {{literal operator cannot have a default argument}}
#endif
}

namespace cwg1482 { // cwg1482: 3.0
                   // NB: sup 2516, test reused there
#if __cplusplus >= 201103L
template <typename T> struct S {
  typedef char I;
};
enum E2 : S<E2>::I { e };
// since-cxx11-error@-1 {{use of undeclared identifier 'E2'}}
#endif
} // namespace cwg1482

namespace cwg1487 { // cwg1487: 3.3
#if __cplusplus >= 201103L
struct A { // #cwg1482-A
  struct B {
    using A::A;
    // since-cxx11-error@-1 {{using declaration refers into 'A::', which is not a base class of 'B'}}
  };

  struct C : A {
  // since-cxx11-error@-1 {{base class has incomplete type}}
  //   since-cxx11-note@#cwg1482-A {{definition of 'cwg1487::A' is not complete until the closing '}'}}
    using A::A;
    // since-cxx11-error@-1 {{using declaration refers into 'A::', which is not a base class of 'C'}}
  };

  struct D;
};

struct D : A {
  using A::A;
};
#endif
} // namespace cwg1487

namespace cwg1490 {  // cwg1490: 3.7 c++11
#if __cplusplus >= 201103L
  // List-initialization from a string literal

  char s[4]{"abc"}; // Ok
  std::initializer_list<char>{"abc"};
  // since-cxx11-error@-1 {{expected unqualified-id}}}
#endif
} // cwg1490

namespace cwg1495 { // cwg1495: 4
#if __cplusplus >= 201103L
  // Deduction succeeds in both directions.
  template<typename T, typename U> struct A {}; // #cwg1495-A
  template<typename T, typename U> struct A<U, T> {};
  // since-cxx11-error@-1 {{class template partial specialization is not more specialized than the primary template}}
  //   since-cxx11-note@#cwg1495-A {{template is declared here}}

  // Primary template is more specialized.
  template<typename, typename...> struct B {}; // #cwg1495-B
  template<typename ...Ts> struct B<Ts...> {};
  // since-cxx11-error@-1 {{class template partial specialization is not more specialized than the primary template}}
  //   since-cxx11-note@#cwg1495-B {{template is declared here}}

  // Deduction fails in both directions.
  template<int, typename, typename ...> struct C {}; // #cwg1495-C
  template<typename ...Ts> struct C<0, Ts...> {};
  // since-cxx11-error@-1 {{class template partial specialization is not more specialized than the primary template}}
  //   since-cxx11-note@#cwg1495-C {{template is declared here}}

#if __cplusplus >= 201402L
  // Deduction succeeds in both directions.
  template<typename T, typename U> int a; // #cwg1495-a
  template<typename T, typename U> int a<U, T>;
  // since-cxx14-error@-1 {{variable template partial specialization is not more specialized than the primary template}}
  //   since-cxx14-note@#cwg1495-a {{template is declared here}}

  // Primary template is more specialized.
  template<typename, typename...> int b; // #cwg1495-b
  template<typename ...Ts> int b<Ts...>;
  // since-cxx14-error@-1 {{variable template partial specialization is not more specialized than the primary template}}
  //   since-cxx14-note@#cwg1495-b {{template is declared here}}

  // Deduction fails in both directions.
  template<int, typename, typename ...> int c; // #cwg1495-c
  template<typename ...Ts> int c<0, Ts...>;
  // since-cxx14-error@-1 {{variable template partial specialization is not more specialized than the primary template}}
  //   since-cxx14-note@#cwg1495-c {{template is declared here}}
#endif
#endif
}

namespace cwg1496 { // cwg1496: no
#if __cplusplus >= 201103L
struct A {
    A() = delete;
};
// FIXME: 'A' should not be trivial because the class lacks at least one
// default constructor which is not deleted.
static_assert(__is_trivial(A), "");
#endif
}
