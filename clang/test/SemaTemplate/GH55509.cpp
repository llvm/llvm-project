// RUN: %clang_cc1 -fsyntax-only -verify -std=c++26 %s

namespace t1 {
  template<int N> struct A {
    template<class C> friend auto cica(const A<N-1>&, C) {
      return N;
    }
  };

  template<> struct A<0> {
    template<class C> friend auto cica(const A<0>&, C);
    // expected-note@-1 {{declared here}}
  };

  void test() {
    cica(A<0>{}, 0);
    // expected-error@-1 {{function 'cica<int>' with deduced return type cannot be used before it is defined}}

    (void)A<1>{};
    cica(A<0>{}, 0);
  }
} // namespace t1
namespace t2 {
  template<int N> struct A {
    template<class C> friend auto cica(const A<N-1>&, C) {
      return N;
    }
  };

  template<> struct A<0> {
    template<class C> friend auto cica(const A<0>&, C);
  };

  template <int N, class = decltype(cica(A<N>{}, nullptr))>
  void MakeCica();
  // expected-note@-1 {{candidate function}}

  template <int N> void MakeCica(A<N+1> = {});
  // expected-note@-1 {{candidate function}}

  void test() {
    MakeCica<0>();

    MakeCica<0>();
    // expected-error@-1 {{call to 'MakeCica' is ambiguous}}
  }
} // namespace t2
namespace t3 {
  template<int N> struct A {
    template<class C> friend auto cica(const A<N-1>&, C) {
      return N-1;
    }
  };

  template<> struct A<0> {
    template<class C> friend auto cica(const A<0>&, C);
  };

  template <int N, class AT, class = decltype(cica(AT{}, nullptr))>
  static constexpr bool MakeCica(int);

  template <int N, class AT>
  static constexpr bool MakeCica(short, A<N+1> = {});

  template <int N, class AT = A<N>, class Val = decltype(MakeCica<N, AT>(0))>
  static constexpr bool has_cica = Val{};

  constexpr bool cica2 = has_cica<0> || has_cica<0>;
} // namespace t3
namespace t4 {
  template<int N> struct A {
    template<class C> friend auto cica(const A<N-1>&, C);
  };

  template<> struct A<0> {
    template<class C> friend auto cica(const A<0>&, C) {
      C a;
    }
  };

  template struct A<1>;

  void test() {
    cica(A<0>{}, 0);
  }
} // namespace t4
namespace regression1 {
  template <class> class A;

  template <class T> [[gnu::abi_tag("TAG")]] void foo(A<T>);

  template <class> struct A {
    friend void foo <>(A);
  };

  template struct A<int>;

  template <class T> [[gnu::abi_tag("TAG")]] void foo(A<T>) {}

  template void foo<int>(A<int>);
} // namespace regression1
