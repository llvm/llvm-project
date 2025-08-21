// RUN: %clang_cc1 -std=c++17 -verify=expected,cxx17 %s
// RUN: %clang_cc1 -std=c++20 -verify=expected,cxx20 %s

namespace pr41427 {
  template <typename T> class A {
  public:
    A(void (*)(T)) {}
  };

  void D(int) {}

  void f() {
    A a(&D);
    using T = decltype(a);
    using T = A<int>;
  }
}

namespace Access {
  struct B {
  protected:
    struct type {};
  };
  template<typename T> struct D : B { // expected-note {{not viable}} \
                                         expected-note {{implicit deduction guide declared as 'template <typename T> D(Access::D<T>) -> Access::D<T>'}}
    D(T, typename T::type); // expected-note {{private member}} \
                            // expected-note {{implicit deduction guide declared as 'template <typename T> D(T, typename T::type) -> Access::D<T>'}}
  };
  D b = {B(), {}};

  class X {
    using type = int;
  };
  D x = {X(), {}}; // expected-error {{no viable constructor or deduction guide}}

  // Once we implement proper support for dependent nested name specifiers in
  // friends, this should still work.
  class Y {
    template <typename T> friend D<T>::D(T, typename T::type); // expected-warning {{dependent nested name specifier}}
    struct type {};
  };
  D y = {Y(), {}};

  class Z {
    template <typename T> friend class D;
    struct type {};
  };
  D z = {Z(), {}};
}

namespace GH69987 {
template<class> struct X {};
template<class = void> struct X;
X x;

template<class T, class B> struct Y { Y(T); };
template<class T, class B=void> struct Y ;
Y y(1);
}

namespace NoCrashOnGettingDefaultArgLoc {
template <typename>
class A {
  A(int = 1); // expected-note {{candidate template ignored: couldn't infer template argumen}} \
              // expected-note {{implicit deduction guide declared as 'template <typename> D(int = <null expr>) -> NoCrashOnGettingDefaultArgLoc::D<type-parameter-0-0>'}}
};
class C : A<int> {
  using A::A;
};
template <typename>
class D : C { // expected-note {{candidate function template not viable: requires 1 argument}} \
                 expected-note {{implicit deduction guide declared as 'template <typename> D(NoCrashOnGettingDefaultArgLoc::D<type-parameter-0-0>) -> NoCrashOnGettingDefaultArgLoc::D<type-parameter-0-0>'}}
  using C::C;
};
D abc; // expected-error {{no viable constructor or deduction guide}}
}

namespace AsValueParameter {
  namespace foo {
    // cxx17-note@+2 {{template is declared here}}
    // cxx20-note@+1 {{'A<int>' is not literal because it is not an aggregate and has no constexpr constructors other than copy or move constructors}}
    template <class> struct A {
      A();
    };
  }
  template <foo::A> struct B {}; // expected-note {{template parameter is declared here}}
  // cxx17-error@-1 {{use of class template 'foo::A' requires template arguments; argument deduction not allowed in template parameter}}

  template struct B<foo::A<int>{}>;
  // cxx17-error@-1 {{value of type 'foo::A<int>' is not implicitly convertible to 'int'}}
  // cxx20-error@-2 {{non-type template parameter has non-literal type 'foo::A<int>' (aka 'AsValueParameter::foo::A<int>')}}
} // namespace AsValueParameter

namespace ConvertDeducedTemplateArgument {
  namespace A {
    template <class> struct B {};
  }

  template <template <class> class TT1> struct C {
    C(TT1<int>);
  };

  template <template <class> class TT2> using D = TT2<int>;

  auto x = C(D<A::B>());
}
