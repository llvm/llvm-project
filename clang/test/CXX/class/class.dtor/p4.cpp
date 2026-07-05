// RUN: %clang_cc1 -std=c++20 -verify %s

template <int N>
struct A {
  ~A() = delete;                  // expected-note {{explicitly marked deleted}}
  ~A() requires(N == 1) = delete; // expected-note {{explicitly marked deleted}}
};

// FIXME: We should probably make it illegal to mix virtual and non-virtual methods
// this way. See CWG2488 and some discussion in https://gcc.gnu.org/bugzilla/show_bug.cgi?id=105699.
template <int N>
struct B {
  ~B() requires(N == 1) = delete; // expected-note {{explicitly marked deleted}}
  virtual ~B() = delete;          // expected-note {{explicitly marked deleted}}
};

template <int N>
concept CO1 = N == 1;

template <int N>
concept CO2 = N >
0;

template <int N>
struct C {
  ~C() = delete; // expected-note {{explicitly marked deleted}}
  ~C() requires(CO1<N>) = delete;
  ~C() requires(CO1<N> &&CO2<N>) = delete; // expected-note {{explicitly marked deleted}}
};

template <int N>
struct D {
  ~D() requires(N != 0) = delete; // expected-note {{explicitly marked deleted}}
  // expected-note@-1 {{candidate function has been explicitly deleted}}
  // expected-note@-2 {{candidate function not viable: constraints not satisfied}}
  // expected-note@-3 {{evaluated to false}}
  ~D() requires(N == 1) = delete;
  // expected-note@-1 {{candidate function has been explicitly deleted}}
  // expected-note@-2 {{candidate function not viable: constraints not satisfied}}
  // expected-note@-3 {{evaluated to false}}
};

template <class T>
concept Foo = requires(T t) {
  {t.foo()};
};

template <int N>
struct E {
  void foo();
  ~E();
  ~E() requires Foo<E> = delete; // expected-note {{explicitly marked deleted}}
};

template struct A<1>;
template struct A<2>;
template struct B<1>;
template struct B<2>;
template struct C<1>;
template struct C<2>;
template struct D<0>; // expected-error {{no viable destructor found for class 'D<0>'}} expected-note {{in instantiation of template}}
template struct D<1>; // expected-error {{destructor of class 'D<1>' is ambiguous}} expected-note {{in instantiation of template}}
template struct D<2>;
template struct E<1>;

int main() {
  A<1> a1; // expected-error {{attempt to use a deleted function}}
  A<2> a2; // expected-error {{attempt to use a deleted function}}
  B<1> b1; // expected-error {{attempt to use a deleted function}}
  B<2> b2; // expected-error {{attempt to use a deleted function}}
  C<1> c1; // expected-error {{attempt to use a deleted function}}
  C<2> c2; // expected-error {{attempt to use a deleted function}}
  D<0> d0;
  D<1> d1;
  D<2> d2; // expected-error {{attempt to use a deleted function}}
  E<1> e1; // expected-error {{attempt to use a deleted function}}
}
