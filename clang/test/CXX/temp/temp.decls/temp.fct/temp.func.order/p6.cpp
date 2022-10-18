// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 %s

template <typename> constexpr bool True = true;
template <typename T> concept C = True<T>;
template <typename T> concept D = C<T> && sizeof(T) > 2;
template <typename T> concept E = D<T> && alignof(T) > 1;

struct A {};
template <typename, auto, int, A, typename...> struct S {};
template <typename, auto, int, A, auto...> struct S2 {};
template <typename T, typename U> struct X {};

namespace p6 {

struct B;

void f(C auto &, auto &) = delete;
template <C Q> void f(Q &, C auto &);

void g(struct A *ap, struct B *bp) {
  f(*ap, *bp);
}

#if 0
// FIXME: [temp.func.order]p6.2.1 is not implemented, matching GCC.
template <typename T, C U, typename V> bool operator==(X<T, U>, V) = delete;
template <C T, C U, C V>               bool operator==(T, X<U, V>);

bool h() {
  return X<void *, int>{} == 0;
}
#endif

template<C T, C auto M, int W, A S,
         template<typename, auto, int, A, typename...> class U,
         typename... Z>
void foo(T, U<T, M, W, S, Z...>) = delete;
template<C T, D auto M, int W, A S,
         template<typename, auto, int, A, typename...> class U,
         typename... Z>
void foo(T, U<T, M, W, S, Z...>) = delete;
template<C T, E auto M, int W, A S,
         template<typename, auto, int, A, typename...> class U,
         typename... Z>
void foo(T, U<T, M, W, S, Z...>);

// check auto template parameter pack.
template<C T, auto M, int W, A S,
         template<typename, auto, int, A, auto...> class U,
         C auto... Z>
void foo2(T, U<T, M, W, S, Z...>) = delete;
template<C T, auto M, int W, A S,
         template<typename, auto, int, A, auto...> class U,
         D auto... Z>
void foo2(T, U<T, M, W, S, Z...>) = delete;
template<C T, auto M, int W, A S,
         template<typename, auto, int, A, auto...> class U,
         E auto... Z>
void foo2(T, U<T, M, W, S, Z...>);

void bar(S<int, 1, 1, A{}, int> s, S2<int, 1, 1, A{}, 0, 0u> s2) {
  foo(0, s);
  foo2(0, s2);
}

template<C auto... T> void bar2();
template<D auto... T> void bar2() = delete;

} // namespace p6

namespace TestConversionFunction {
struct Y {
  template<C        T, typename U> operator X<T, U>(); // expected-note {{candidate function [with T = int, U = int]}}
  template<typename T, typename U> operator X<U, T>(); // expected-note {{candidate function [with T = int, U = int]}}
};

X<int,int> f() {
  return Y{}; // expected-error {{conversion from 'Y' to 'X<int, int>' is ambiguous}}
}
}

namespace ClassPartialSpecPartialOrdering {
template<D T> struct Y { Y()=delete; }; // expected-note {{template is declared here}}
template<C T> struct Y<T> {}; // expected-error {{class template partial specialization is not more specialized than the primary template}}

template<C T, int I> struct Y1 { Y1()=delete; };
template<D T> struct Y1<T, 2>  { Y1()=delete; };
template<E T> struct Y1<T, 1+1> {};

template<class T, int I, int U> struct Y2 {};
template<class T, int I> struct Y2<T*, I, I+2> {}; // expected-note {{partial specialization matches}}
template<C     T, int I> struct Y2<T*, I, I+1+1> {}; // expected-note {{partial specialization matches}}

template<C T, C auto I, int W, A S, template<typename, auto, int, A, typename...> class U, typename... Z>
struct Y3 { Y3()=delete; };
template<C T, D auto I, int W, A S, template<typename, auto, int, A, typename...> class U, typename... Z>
struct Y3<T, I, W, S, U, Z...> { Y3()=delete; };
template<C T, E auto I, int W, A S, template<typename, auto, int, A, typename...> class U, typename... Z>
struct Y3<T, I, W, S, U, Z...> {};

void f() {
  Y1<int, 2> a;
  Y2<char*, 1, 3> b; // expected-error {{ambiguous partial specializations}}
  Y3<int, 1, 1, A{}, S, int> c;
}

template<C T, C V> struct Y4; // expected-note {{template is declared here}}
template<D T, C V> struct Y4<V, T>; // expected-error {{class template partial specialization is not more specialized than the primary template}}

template<C auto T> struct W1;
template<D auto T> struct W1<T> {};

template<C auto... T> struct W2;
template<D auto... T> struct W2<T...> {};

template<class T, class U>
concept C1 = C<T> && C<U>;
template<class T, class U>
concept D1 = D<T> && C<U>;

template<C1<A> auto T> struct W3;
template<D1<A> auto T> struct W3<T> {};

template<C1<A> auto... T> struct W4;
template<D1<A> auto... T> struct W4<T...> {};

// FIXME: enable once Clang support non-trivial auto on NTTP.
// template<C auto* T> struct W5;
// template<D auto* T> struct W5<T> {};

// FIXME: enable once Clang support non-trivial auto on NTTP.
// template<C auto& T> struct W6;
// template<D auto& T> struct W6<T> {};

struct W1<0> w1;
struct W2<0> w2;
struct W3<0> w3;
struct W4<0> w4;
// FIXME: enable once Clang support non-trivial auto on NTTP.
// struct W5<(int*)nullptr> w5;
// struct W6<w5> w6;
}

namespace PR53640 {

template <typename T>
concept C = true;

template <C T>
void f(T t) {} // expected-note {{candidate function [with T = int]}}

template <typename T>
void f(const T &t) {} // expected-note {{candidate function [with T = int]}}

int g() {
  f(0); // expected-error {{call to 'f' is ambiguous}}
}

struct S {
  template <typename T> explicit S(T) noexcept requires C<T> {} // expected-note {{candidate constructor}}
  template <typename T> explicit S(const T &) noexcept {}       // expected-note {{candidate constructor}}
};

int h() {
  S s(4); // expected-error-re {{call to constructor of {{.*}} is ambiguous}}
}

}
