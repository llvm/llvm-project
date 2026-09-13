// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s
// expected-no-diagnostics
//
// Regression test for https://github.com/llvm/llvm-project/issues/223220.
//
// Mechanism: the constraint of a function template is checked for one
// specialization while a constraint check of *another* specialization of the
// same function template is still active. Both specializations share the same
// original parameter declarations. When the parameters of the inner
// specialization are looked up through the (outer) instantiation scope chain,
// the parameter mapping of the outer specialization can be found and reused.
// The inner constraint expression then ends up with parameters of the outer
// specialization's type, which can re-enter the very same constraint check and
// be reported as "satisfaction of constraint ... depends on itself".
//
// This test reproduces the structure of std::__detail::__synth3way_t
// instantiated from the operator<=> of std::map and std::pair (no standard
// library involved):
//   Map::operator<=>  -> synth<Pair<const Key, Node>>   (outer check)
//     `a < b` inside the outer check's requirement visits Pair's rewritten
//     operator<=>, whose return type instantiates synth<const Key, Key>
//     (the inner check) while the outer check is still active.

namespace nested_synth {

template <class T> T &&Declval() noexcept;

struct Synth {
  template <class T, class U>
  constexpr auto operator()(T const &t, U const &u) const
      requires requires {
        { t < u };
        { u < t };
      } {
    if (t < u)
      return int{-1};
    if (u < t)
      return int{1};
    return int{0};
  }
};
constexpr Synth synth{};

template <class T, class U = T>
using synth_t = decltype(synth(Declval<T &>(), Declval<U &>()));

struct Key {
  int n;
  constexpr bool operator<(Key const &O) const { return n < O.n; }
};
struct Node {
  int m;
  constexpr bool operator<(Node const &O) const { return m < O.m; }
};

template <class T1, class T2> struct Pair {
  T1 first;
  T2 second;
};
// Pair has no operator<, so `lhs < rhs` resolves through the rewritten
// operator<=>, whose return type instantiates synth_t<T1, U1>.
template <class T1, class T2, class U1, class U2>
constexpr auto operator<=>(Pair<T1, T2> const &, Pair<U1, U2> const &)
    -> synth_t<T1, U1> {
  return {};
}

template <class K2, class V2> struct Map {};

// The map's only ordering operator returns the synthesized three-way result
// of its (const Key, V) element pairs.
template <class K2, class V2>
constexpr auto operator<=>(Map<K2, V2> const &, Map<K2, V2> const &)
    -> synth_t<Pair<const K2, V2>> {
  return {};
}

using NodeMap = Map<Key, Node>;

bool compare(NodeMap const &A, NodeMap const &B) { return A < B; }

} // namespace nested_synth
