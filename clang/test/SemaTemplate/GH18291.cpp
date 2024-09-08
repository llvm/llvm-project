// RUN: %clang_cc1 -std=c++23 -verify %s

namespace t1 {
  template<bool> struct enable_if { typedef void type; };
  template <class T> class Foo {};
  template <class X> constexpr bool check() { return true; }
  template <class X, class Enable = void> struct Bar {};

  namespace param {
    template<class X> void func(Bar<X, typename enable_if<check<X>()>::type>) {}
    // expected-note@-1 2{{candidate function}}
    template<class T> void func(Bar<Foo<T>>) {}
    // expected-note@-1 2{{candidate function}}

    void g() {
      func(Bar<Foo<int>>()); // expected-error {{call to 'func' is ambiguous}}
      void (*ptr)(Bar<Foo<int>>){func};
      // expected-error@-1 {{address of overloaded function 'func' is ambiguous}}
    }
  } // namespace param
  namespace ret {
    template<class X> Bar<X, typename enable_if<check<X>()>::type> func();
    // expected-note@-1 {{candidate function}}
    template<class T> Bar<Foo<T>> func();
    // expected-note@-1 {{candidate function}}

    void g() {
      Bar<Foo<int>> (*ptr)(){func};
      // expected-error@-1 {{address of overloaded function 'func' is ambiguous}}
    }
  } // namespace ret
  namespace conv {
    struct A {
      template<class X> operator Bar<X, typename enable_if<check<X>()>::type>();
      // expected-note@-1 {{candidate function}}
      template<class T> operator Bar<Foo<T>>();
      // expected-note@-1 {{candidate function}}
    };
    void g() {
      Bar<Foo<int>> x = A();
      // expected-error@-1 {{conversion from 'A' to 'Bar<Foo<int>>' is ambiguous}}
    }
  } // namespace conv
} // namespace t1

namespace t2 {
  template <bool> struct enable_if;
  template <> struct enable_if<true> {
    typedef int type;
  };
  struct pair {
    template <int = 0> pair(int);
    template <class _U2, enable_if<__is_constructible(int &, _U2)>::type = 0>
    pair(_U2 &&);
  };
  int test_test_i;
  void test() { pair{test_test_i}; }
} // namespace t2

namespace t3 {
  template <class _Tp> void to_address(_Tp);
  template <class _Pointer> auto to_address(_Pointer __p) -> decltype(__p);

  template <class _CharT> struct basic_string_view {
    basic_string_view(_CharT);

    template <class _It> requires requires(_It __i) { to_address(__i); }
    basic_string_view(_It);
  };
  void operatorsv() { basic_string_view(0); }
} // namespace t3

namespace func_pointer {
  template <class> struct __promote {
    using type = float;
  };
  template <class> class complex {};

  namespace ret {
    template <class _Tp> complex<_Tp> pow(const complex<_Tp> &) {};
    template <class _Tp> complex<typename __promote<_Tp>::type> pow(_Tp) = delete;
    complex<float> (*ptr)(const complex<float> &){pow};
  } // namespace ret
  namespace param {
    template <class _Tp> void pow(const complex<_Tp> &, complex<_Tp>) {};
    template <class _Tp> void pow(_Tp, complex<typename __promote<_Tp>::type>) = delete;
    void (*ptr)(const complex<float> &, complex<float>){pow};
  } // namespace param
} // namespace t3
