// RUN: %clang_analyze_cc1 -Wno-ignored-reference-qualifiers -analyzer-checker=core,debug.ExprInspection -std=c++17 -verify %s

#include "Inputs/system-header-simulator-cxx.h"

void clang_analyzer_eval(bool);

namespace std {
template <typename T>
struct tuple_size {
};

template <std::size_t I, typename T>
struct tuple_element {
};

// The std::pair in our system header simulator is not tuple-like, so a tuple-like mock is created here
template <typename T1, typename T2>
struct mock_pair {
  T1 first;
  T2 second;
};
template <typename T1, typename T2>
struct tuple_size<mock_pair<T1, T2>> {
  static const std::size_t value = 2;
};

template <typename T1, typename T2>
struct tuple_element<0, mock_pair<T1, T2>> {
  using type = T1;
};

template <typename T1, typename T2>
struct tuple_element<1, mock_pair<T1, T2>> {
  using type = T2;
};

template <std::size_t I, class T>
using tuple_element_t = typename tuple_element<I, T>::type;

template <std::size_t I, class T1, class T2>
constexpr std::tuple_element_t<I, std::mock_pair<T1, T2>> &
get(std::mock_pair<T1, T2> &p) noexcept {
  if (I == 0)
    return p.first;
  else
    return p.second;
}

template <std::size_t I, class T1, class T2>
constexpr const std::tuple_element_t<I, std::mock_pair<T1, T2>> &
get(const std::mock_pair<T1, T2> &p) noexcept {
  if (I == 0)
    return p.first;
  else
    return p.second;
}

template <std::size_t I, class T1, class T2>
constexpr std::tuple_element_t<I, std::mock_pair<T1, T2>> &&
get(std::mock_pair<T1, T2> &&p) noexcept {

  if (I == 0)
    return static_cast<std::tuple_element_t<I, std::mock_pair<T1, T2>> &&>(p.first);
  else
    return static_cast<std::tuple_element_t<I, std::mock_pair<T1, T2>> &&>(p.second);
}

template <std::size_t I, class T1, class T2>
constexpr const std::tuple_element_t<I, std::mock_pair<T1, T2>> &&
get(const std::mock_pair<T1, T2> &&p) noexcept {
  if (I == 0)
    return static_cast<std::tuple_element_t<I, std::mock_pair<T1, T2>> &&>(p.first);
  else
    return static_cast<std::tuple_element_t<I, std::mock_pair<T1, T2>> &&>(p.second);
}

} // namespace std
// A utility that generates a tuple-like struct with 2 fields
//  of the same type. The fields are 'first' and 'second'
#define GENERATE_TUPLE_LIKE_STRUCT(name, element_type) \
  struct name {                                        \
    element_type first;                                \
    element_type second;                               \
  };                                                   \
                                                       \
  namespace std {                                      \
  template <>                                          \
  struct tuple_size<name> {                            \
    static const std::size_t value = 2;                \
  };                                                   \
                                                       \
  template <std::size_t I>                             \
  struct tuple_element<I, name> {                      \
    using type = element_type;                         \
  };                                                   \
  }

void non_user_defined_by_value(void) {
  std::mock_pair<int, int> p = {1, 2};

  auto [u, v] = p;

  clang_analyzer_eval(u == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(v == 2); // expected-warning{{TRUE}}

  int x = u;
  u = 10;
  int y = u;

  clang_analyzer_eval(x == 1);  // expected-warning{{TRUE}}
  clang_analyzer_eval(u == 10); // expected-warning{{TRUE}}

  clang_analyzer_eval(y == 10);      // expected-warning{{TRUE}}
  clang_analyzer_eval(p.first == 1); // expected-warning{{TRUE}}

  p.first = 5;

  clang_analyzer_eval(u == 10); // expected-warning{{TRUE}}
}

void non_user_defined_by_lref(void) {
  std::mock_pair<int, int> p = {1, 2};

  auto &[u, v] = p;

  int x = u;
  u = 10;
  int y = u;

  clang_analyzer_eval(x == 1);  // expected-warning{{TRUE}}
  clang_analyzer_eval(u == 10); // expected-warning{{TRUE}}

  clang_analyzer_eval(y == 10);       // expected-warning{{TRUE}}
  clang_analyzer_eval(p.first == 10); // expected-warning{{TRUE}}

  clang_analyzer_eval(v == 2);        // expected-warning{{TRUE}}
  clang_analyzer_eval(p.second == 2); // expected-warning{{TRUE}}

  p.first = 5;

  clang_analyzer_eval(u == 5); // expected-warning{{TRUE}}
}

void non_user_defined_by_rref(void) {
  std::mock_pair<int, int> p = {1, 2};

  auto &&[u, v] = p;

  int x = u;
  u = 10;
  int y = u;

  clang_analyzer_eval(x == 1);  // expected-warning{{TRUE}}
  clang_analyzer_eval(u == 10); // expected-warning{{TRUE}}

  clang_analyzer_eval(y == 10);       // expected-warning{{TRUE}}
  clang_analyzer_eval(p.first == 10); // expected-warning{{TRUE}}

  clang_analyzer_eval(v == 2);        // expected-warning{{TRUE}}
  clang_analyzer_eval(p.second == 2); // expected-warning{{TRUE}}

  p.first = 5;

  clang_analyzer_eval(u == 5); // expected-warning{{TRUE}}
}

GENERATE_TUPLE_LIKE_STRUCT(Test, int);

template <std::size_t I>
int get(Test t) {
  if (I == 0) {
    t.second = 10;
    return t.first;
  } else {
    t.first = 20;
    return t.second;
  }
}

void user_defined_get_val_by_val(void) {
  Test p{1, 2};
  auto [u, v] = p;

  clang_analyzer_eval(u == 1); // expected-warning{{TRUE}}

  u = 8;

  int x = u;

  clang_analyzer_eval(x == 8); // expected-warning{{TRUE}}

  clang_analyzer_eval(u == 8); // expected-warning{{TRUE}}
  clang_analyzer_eval(v == 2); // expected-warning{{TRUE}}

  clang_analyzer_eval(p.first == 1);  // expected-warning{{TRUE}}
  clang_analyzer_eval(p.second == 2); // expected-warning{{TRUE}}

  p.first = 5;

  clang_analyzer_eval(u == 8);       // expected-warning{{TRUE}}
  clang_analyzer_eval(p.first == 5); // expected-warning{{TRUE}}
}

GENERATE_TUPLE_LIKE_STRUCT(Test2, int);

template <std::size_t I>
int get(Test2 &t) {
  if (I == 0) {
    t.second = 10;
    return t.first;
  } else {
    t.first = 20;
    return t.second;
  }
}

void user_defined_get_val_by_lref(void) {
  Test2 p{1, 2};

  auto &[u, v] = p;

  clang_analyzer_eval(u == 1);  // expected-warning{{TRUE}}
  clang_analyzer_eval(v == 10); // expected-warning{{TRUE}}

  u = 8;

  int x = u;

  clang_analyzer_eval(x == 8); // expected-warning{{TRUE}}

  clang_analyzer_eval(u == 8);  // expected-warning{{TRUE}}
  clang_analyzer_eval(v == 10); // expected-warning{{TRUE}}

  clang_analyzer_eval(p.first == 20);  // expected-warning{{TRUE}}
  clang_analyzer_eval(p.second == 10); // expected-warning{{TRUE}}

  p.first = 5;

  clang_analyzer_eval(u == 8);       // expected-warning{{TRUE}}
  clang_analyzer_eval(p.first == 5); // expected-warning{{TRUE}}
}

void user_defined_get_val_by_rref(void) {
  Test2 p{1, 2};

  auto &&[u, v] = p;

  clang_analyzer_eval(u == 1);  // expected-warning{{TRUE}}
  clang_analyzer_eval(v == 10); // expected-warning{{TRUE}}

  u = 8;

  int x = u;

  clang_analyzer_eval(x == 8); // expected-warning{{TRUE}}

  clang_analyzer_eval(u == 8);  // expected-warning{{TRUE}}
  clang_analyzer_eval(v == 10); // expected-warning{{TRUE}}

  clang_analyzer_eval(p.first == 20);  // expected-warning{{TRUE}}
  clang_analyzer_eval(p.second == 10); // expected-warning{{TRUE}}

  p.first = 5;

  clang_analyzer_eval(u == 8);       // expected-warning{{TRUE}}
  clang_analyzer_eval(p.first == 5); // expected-warning{{TRUE}}
}

struct MixedTest {
  int x;
  char &&y;
  int &z;
};

namespace std {
template <>
struct tuple_size<MixedTest> {
  static const std::size_t value = 3;
};

template <>
struct tuple_element<0, MixedTest> {
  using type = int;
};

template <>
struct tuple_element<1, MixedTest> {
  using type = char &&;
};

template <>
struct tuple_element<2, MixedTest> {
  using type = int &;
};

template <std::size_t I, typename T>
using tuple_element_t = typename tuple_element<I, T>::type;

} // namespace std

template <std::size_t I>
const std::tuple_element_t<I, MixedTest> &get(const MixedTest &t) {}

template <>
const std::tuple_element_t<0, MixedTest> &get<0>(const MixedTest &t) {
  return t.x;
}

template <>
const std::tuple_element_t<1, MixedTest> &get<1>(const MixedTest &t) {
  return t.y;
}

template <>
const std::tuple_element_t<2, MixedTest> &get<2>(const MixedTest &t) {
  return t.z;
}

void mixed_type_cref(void) {
  int x = 1;
  char y = 2;
  int z = 3;

  MixedTest m{x, std::move(y), z};
  const auto &[a, b, c] = m;

  clang_analyzer_eval(a == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(b == 2); // expected-warning{{TRUE}}
  clang_analyzer_eval(c == 3); // expected-warning{{TRUE}}

  clang_analyzer_eval(a == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(b == 2); // expected-warning{{TRUE}}
  clang_analyzer_eval(c == 3); // expected-warning{{TRUE}}
}

template <std::size_t I>
std::tuple_element_t<I, MixedTest> &get(MixedTest &t) {}

template <>
std::tuple_element_t<0, MixedTest> &get<0>(MixedTest &t) {
  return t.x;
}

template <>
std::tuple_element_t<1, MixedTest> &get<1>(MixedTest &t) {
  return t.y;
}

template <>
std::tuple_element_t<2, MixedTest> &get<2>(MixedTest &t) {
  return t.z;
}

void mixed_type_lref(void) {
  int x = 1;
  char y = 2;
  int z = 3;

  MixedTest m{x, std::move(y), z};
  auto &[a, b, c] = m;

  a = 4;
  b = 5;
  c = 6;

  clang_analyzer_eval(get<0>(m) == 4); // expected-warning{{TRUE}}
  clang_analyzer_eval(get<1>(m) == 5); // expected-warning{{TRUE}}
  clang_analyzer_eval(get<2>(m) == 6); // expected-warning{{TRUE}}

  clang_analyzer_eval(get<0>(m) == 4); // expected-warning{{TRUE}}
  clang_analyzer_eval(get<1>(m) == 5); // expected-warning{{TRUE}}
  clang_analyzer_eval(get<2>(m) == 6); // expected-warning{{TRUE}}

  clang_analyzer_eval(z == 6); // expected-warning{{TRUE}}
}

void mixed_type_rref(void) {
  int x = 1;
  char y = 2;
  int z = 3;

  MixedTest m{x, std::move(y), z};
  auto &&[a, b, c] = m;

  a = 4;
  b = 5;
  c = 6;

  clang_analyzer_eval(get<0>(m) == 4); // expected-warning{{TRUE}}
  clang_analyzer_eval(get<1>(m) == 5); // expected-warning{{TRUE}}
  clang_analyzer_eval(get<2>(m) == 6); // expected-warning{{TRUE}}

  clang_analyzer_eval(get<0>(m) == 4); // expected-warning{{TRUE}}
  clang_analyzer_eval(get<1>(m) == 5); // expected-warning{{TRUE}}
  clang_analyzer_eval(get<2>(m) == 6); // expected-warning{{TRUE}}

  clang_analyzer_eval(z == 6); // expected-warning{{TRUE}}
}

void ref_val(void) {
  int i = 1, j = 2;
  std::mock_pair<int &, int &> p{i, j};

  auto [a, b] = p;
  clang_analyzer_eval(a == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(b == 2); // expected-warning{{TRUE}}

  a = 3;
  b = 4;

  clang_analyzer_eval(p.first == 3);  // expected-warning{{TRUE}}
  clang_analyzer_eval(p.second == 4); // expected-warning{{TRUE}}

  clang_analyzer_eval(a == 3); // expected-warning{{TRUE}}
  clang_analyzer_eval(b == 4); // expected-warning{{TRUE}}
}

struct Small_Non_POD {
  int i;
  int j;
};

void non_user_defined_small_non_pod_by_value(void) {
  std::mock_pair<Small_Non_POD, Small_Non_POD> p{{1, 2}, {1, 2}};

  auto [a, b] = p;

  clang_analyzer_eval(a.i == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(a.j == 2); // expected-warning{{TRUE}}

  clang_analyzer_eval(b.i == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(b.j == 2); // expected-warning{{TRUE}}

  a.i = 3;
  a.j = 4;

  b.i = 5;
  b.j = 6;

  clang_analyzer_eval(a.i == 3); // expected-warning{{TRUE}}
  clang_analyzer_eval(a.j == 4); // expected-warning{{TRUE}}

  clang_analyzer_eval(b.i == 5); // expected-warning{{TRUE}}
  clang_analyzer_eval(b.j == 6); // expected-warning{{TRUE}}

  clang_analyzer_eval(p.first.i == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(p.first.j == 2); // expected-warning{{TRUE}}

  clang_analyzer_eval(p.second.i == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(p.second.j == 2); // expected-warning{{TRUE}}
}

void non_user_defined_small_non_pod_by_lref(void) {
  std::mock_pair<Small_Non_POD, Small_Non_POD> p{{1, 2}, {1, 2}};

  auto &[a, b] = p;

  clang_analyzer_eval(a.i == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(a.j == 2); // expected-warning{{TRUE}}

  clang_analyzer_eval(b.i == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(b.j == 2); // expected-warning{{TRUE}}

  a.i = 3;
  a.j = 4;

  b.i = 5;
  b.j = 6;

  clang_analyzer_eval(a.i == 3); // expected-warning{{TRUE}}
  clang_analyzer_eval(a.j == 4); // expected-warning{{TRUE}}

  clang_analyzer_eval(b.i == 5); // expected-warning{{TRUE}}
  clang_analyzer_eval(b.j == 6); // expected-warning{{TRUE}}

  clang_analyzer_eval(p.first.i == 3); // expected-warning{{TRUE}}
  clang_analyzer_eval(p.first.j == 4); // expected-warning{{TRUE}}

  clang_analyzer_eval(p.second.i == 5); // expected-warning{{TRUE}}
  clang_analyzer_eval(p.second.j == 6); // expected-warning{{TRUE}}
}

void non_user_defined_small_non_pod_by_rref(void) {
  std::mock_pair<Small_Non_POD, Small_Non_POD> p{{1, 2}, {1, 2}};

  auto &&[a, b] = p;

  clang_analyzer_eval(a.i == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(a.j == 2); // expected-warning{{TRUE}}

  clang_analyzer_eval(b.i == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(b.j == 2); // expected-warning{{TRUE}}

  a.i = 3;
  a.j = 4;

  b.i = 5;
  b.j = 6;

  clang_analyzer_eval(a.i == 3); // expected-warning{{TRUE}}
  clang_analyzer_eval(a.j == 4); // expected-warning{{TRUE}}

  clang_analyzer_eval(b.i == 5); // expected-warning{{TRUE}}
  clang_analyzer_eval(b.j == 6); // expected-warning{{TRUE}}

  clang_analyzer_eval(p.first.i == 3); // expected-warning{{TRUE}}
  clang_analyzer_eval(p.first.j == 4); // expected-warning{{TRUE}}

  clang_analyzer_eval(p.second.i == 5); // expected-warning{{TRUE}}
  clang_analyzer_eval(p.second.j == 6); // expected-warning{{TRUE}}
}

GENERATE_TUPLE_LIKE_STRUCT(Uninit, int);
template <std::size_t I>
int &get(Uninit &&t) {
  if (I == 0) {
    return t.first;
  } else {
    return t.second;
  }
}

void uninit_a(void) {
  Uninit u;

  auto [a, b] = u;

  int x = a; // expected-warning{{Assigned value is garbage or undefined}}
}

void uninit_b(void) {
  Uninit u;

  auto [a, b] = u;

  int x = b; // expected-warning{{Assigned value is garbage or undefined}}
}

GENERATE_TUPLE_LIKE_STRUCT(UninitCall, int);
template <std::size_t I>
int get(UninitCall t) {
  if (I == 0) {
    return t.first;
  } else {
    return t.second;
  }
}

void uninit_call(void) {
  UninitCall u;

  auto [a, b] = u;

  int x = a;
  // expected-warning@543{{Undefined or garbage value returned to caller}}
}

void syntax_2() {
  std::mock_pair<Small_Non_POD, Small_Non_POD> p{{1, 2}, {3, 4}};

  auto [a, b]{p};

  clang_analyzer_eval(a.i == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(a.j == 2); // expected-warning{{TRUE}}

  clang_analyzer_eval(b.i == 3); // expected-warning{{TRUE}}
  clang_analyzer_eval(b.j == 4); // expected-warning{{TRUE}}
}

void syntax_3() {
  std::mock_pair<Small_Non_POD, Small_Non_POD> p{{1, 2}, {3, 4}};

  auto [a, b](p);

  clang_analyzer_eval(a.i == 1); // expected-warning{{TRUE}}
  clang_analyzer_eval(a.j == 2); // expected-warning{{TRUE}}

  clang_analyzer_eval(b.i == 3); // expected-warning{{TRUE}}
  clang_analyzer_eval(b.j == 4); // expected-warning{{TRUE}}
}
