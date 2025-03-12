// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++1z %s

template<typename T> class A; // expected-note 2 {{template parameter is declared here}} expected-note{{template is declared here}}

// [temp.arg.type]p1
A<0> *a1; // expected-error{{template argument for template type parameter must be a type}}

A<A> *a2; // expected-error{{use of class template 'A' requires template arguments}}

A<int> *a3;
A<int()> *a4;
A<int(float)> *a5;
A<A<int> > *a6;

// Pass an overloaded function template:
template<typename T> void function_tpl(T);
A<function_tpl> a7;  // expected-error{{template argument for template type parameter must be a type}}

// Pass a qualified name:
namespace ns {
template<typename T> class B {};  // expected-note{{template is declared here}}
}
A<ns::B> a8; // expected-error{{use of class template 'ns::B' requires template arguments}}

// [temp.arg.type]p2
void f() {
  class X { };
  A<X> * a = 0;
#if __cplusplus <= 199711L
  // expected-warning@-2 {{template argument uses local type 'X'}}
#endif
}

struct { int x; } Unnamed;
#if __cplusplus <= 199711L
// expected-note@-2 {{unnamed type used in template argument was declared here}}
#endif

A<__typeof__(Unnamed)> *a9;
#if __cplusplus <= 199711L
// expected-warning@-2 {{template argument uses unnamed type}}
#endif

template<typename T, unsigned N>
struct Array {
  typedef struct { T x[N]; } type;
};

template<typename T> struct A1 { };
A1<Array<int, 17>::type> ax;

// FIXME: [temp.arg.type]p3. The check doesn't really belong here (it
// belongs somewhere in the template instantiation section).

#if __cplusplus >= 201703
// As a defect resolution, we support deducing B in noexcept(B).
namespace deduce_noexcept {
  template<typename> struct function;
  template<typename R, typename ...A, bool N>
  struct function<R(A...) noexcept(N)> {
    static constexpr bool Noexcept = N;
  };
  static_assert(function<int(float, double) noexcept>::Noexcept);
  static_assert(!function<int(float, double)>::Noexcept);

  void noexcept_function() noexcept;
  void throwing_function();

  template<typename T, bool B> float &deduce_function(T(*)() noexcept(B));
  template<typename T> int &deduce_function(T(*)() noexcept);
  void test_function_deduction() {
    int &r = deduce_function(noexcept_function);
    float &s = deduce_function(throwing_function);
  }

  namespace low_priority_deduction {
    template<int> struct A {};
    template<auto B> void f(A<B>, void(*)() noexcept(B)) {
      using T = decltype(B);
      using T = int;
    }
    void g() { f(A<0>(), g); } // ok, deduce B as an int
  }

  // FIXME: It's not clear whether this should work. We're told to deduce with
  // P being the function template type and A being the declared type, which
  // would accept this, but considering the exception specification in such
  // cases breaks new/delete matching.
  template<bool Noexcept> void dep() noexcept(Noexcept) {} // expected-note 3{{couldn't infer template argument 'Noexcept'}}
  template void dep(); // expected-error {{does not refer to a function template}}
  template void dep() noexcept(true); // expected-error {{does not refer to a function template}}
  template void dep() noexcept(false); // expected-error {{does not refer to a function template}}

  template<typename T> typename T::type1 f() noexcept(T::a); // expected-note {{candidate}}
  template<typename T> typename T::type2 f() noexcept(T::b) {} // expected-note {{candidate}}
  struct X {
    static constexpr bool b = true;
    using type1 = void;
    using type2 = void;
  };
  template void f<X>(); // expected-error {{partial ordering for explicit instantiation of 'f' is ambiguous}}
}
#endif
