// RUN: %clang_cc1 -std=c++1z -verify %s

template<typename T, typename U> constexpr bool same = false;
template<typename T> constexpr bool same<T, T> = true;

auto a() {
  if constexpr (false)
    return 0;
}
static_assert(same<decltype(a()), void>);

auto b() {
  if constexpr (false)
    return 0;
  else
    return 0.0;
}
static_assert(same<decltype(b()), double>);

auto c() {
  if constexpr (true)
    return "foo";
  else
    return 'x';
  if constexpr (false)
    return 7.6;
  else
    return 5; // expected-error {{deduced as 'int' here but deduced as 'const char *' in earlier}}
}

template<int k> auto d() {
  if constexpr(k == 0)
    return 0;
  if constexpr(k == 1)
    return "foo";
  else if constexpr (k == 2)
    return 1.0;
}
static_assert(same<decltype(d<0>()), int>);
static_assert(same<decltype(d<1>()), const char *>);
static_assert(same<decltype(d<2>()), double>);
static_assert(same<decltype(d<3>()), void>);

auto e = []{ if constexpr (false) return 0; }(); // expected-error {{variable has incomplete type 'void'}}

auto f = []{ if constexpr (true) return 0; }();
static_assert(same<decltype(e), int>);
