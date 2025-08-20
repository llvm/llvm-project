// RUN: %clang_cc1 -fsyntax-only -std=c++11 %s

template <class> struct Pair;
template <class...> struct Tuple {
  template <class _Up> Tuple(_Up);
};
template <typename> struct StatusOr;
template <int> using ElementType = int;
template <int... fields>
using Key = Tuple<ElementType<fields>...>;
template <int... fields>
StatusOr<Pair<Key<fields...>>> Parser();
struct Helper { Helper(Tuple<>, Tuple<>, int, int); };
struct D : Helper {
  D(Key<> f, int n, int e) : Helper(f, Parser<>, n, e) {}
};
