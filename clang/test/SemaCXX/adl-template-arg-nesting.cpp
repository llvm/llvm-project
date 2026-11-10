// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

namespace inner {
struct Tag {};
void find_me(...) {}
} // namespace inner

template <class A, class B> struct F {};

// Recursive template uses the previous one 2x, resulting in multiple visits per
// type.
template <int N> struct Chain {
  using type = F<typename Chain<N - 1>::type, typename Chain<N - 1>::type>;
};
template <> struct Chain<0> { using type = inner::Tag; };

void huge() {
  find_me(Chain<40>::type());
}

// Same test for function types.
template <int N> struct FnChain {
  using type = void (*)(typename FnChain<N - 1>::type,
                        typename FnChain<N - 1>::type);
};
template <> struct FnChain<0> { using type = void (*)(inner::Tag); };

void huge_fn(FnChain<40>::type fn) {
  find_me(fn);
}

// Make sure we don't regress when visiting the  base-class of an argument.
namespace ns_x {
struct X {};
void h(...) {}
} // namespace ns_x

template <class T> struct B {};
struct D : B<ns_x::X> {};

void bases(D d, B<ns_x::X> b) {
  h(d, b); // OK: found through B<ns_x::X>, the type of the second argument.
}

void only_base(D d) {
  h(d); // expected-error {{use of undeclared identifier 'h'}}
}
