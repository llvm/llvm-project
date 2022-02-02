// RUN: %clang_cc1 -std=c++20 -verify %s
// RUN: %clang_cc1 -std=c++2b -verify %s

namespace PR52905 {
template <class> concept C = true;

struct A {
  int begin();
  int begin() const;
};

template <class T>
concept Beginable = requires (T t) {
  { t.begin } -> C;
  // expected-note@-1 {{because 't.begin' would be invalid: reference to non-static member function must be called}}
};

static_assert(Beginable<A>); // expected-error {{static_assert failed}}
                             // expected-note@-1 {{does not satisfy 'Beginable'}}
} // namespace PR52905

namespace PR52909a {

template<class> constexpr bool B = true;
template<class T> concept True = B<T>;

template <class T>
int foo(T t) requires requires { // expected-note {{candidate template ignored: constraints not satisfied}}
    {t.begin} -> True; // expected-note {{because 't.begin' would be invalid: reference to non-static member function must be called}}
}
{}

struct A { int begin(); };
auto x = foo(A()); // expected-error {{no matching function for call to 'foo'}}

} // namespace PR52909a

namespace PR52909b {

template<class> concept True = true;

template<class T> concept C = requires {
    { T::begin } -> True; // expected-note {{because 'T::begin' would be invalid: reference to overloaded function could not be resolved}}
};

struct A {
    static void begin(int);
    static void begin(double);
};

static_assert(C<A>); // expected-error {{static_assert failed}}
  // expected-note@-1 {{because 'PR52909b::A' does not satisfy 'C'}}

} // namespace PR52909b

namespace PR53075 {
template<class> concept True = true;

template<class T> concept C = requires {
    { &T::f } -> True; // expected-note {{because '&T::f' would be invalid: reference to overloaded function could not be resolved}}
};

struct S {
    int *f();
    int *f() const;
};

static_assert(C<S>); // expected-error {{static_assert failed}}
  // expected-note@-1 {{because 'PR53075::S' does not satisfy 'C'}}

} // namespace PR53075
