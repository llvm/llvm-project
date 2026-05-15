// RUN: %clang_cc1 -verify -std=c++20 -triple x86_64-pc-linux-gnu -fopenmp \
// RUN: -fsyntax-only %s 

namespace std {
  typedef unsigned long size_t;

  // pair
  template <typename T1, typename T2>
  struct pair {
    T1 first;
    T2 second;
  };

  template <typename T1, typename T2>
  pair<T1, T2> make_pair(T1 a, T2 b) {
    return {a, b};
  }

  // tuple
  template <typename... Ts>
  struct tuple;

  template <typename T, typename... Ts>
  struct tuple<T, Ts...> {
    T head;
    tuple<Ts...> tail;
  };

  template <>
  struct tuple<> {};

  template <size_t I, typename T>
  struct tuple_element;

  template <typename T1, typename T2>
  struct tuple_element<0, pair<T1, T2>> { using type = T1; };

  template <typename T1, typename T2>
  struct tuple_element<1, pair<T1, T2>> { using type = T2; };

  template <typename T, typename... Ts>
  struct tuple_element<0, tuple<T, Ts...>> { using type = T; };

  template <size_t I, typename T, typename... Ts>
  struct tuple_element<I, tuple<T, Ts...>> {
    using type = typename tuple_element<I-1, tuple<Ts...>>::type;
  };

  template <size_t N, typename T>
  struct tuple_element<0, T[N]> { using type = T; };

  template <typename T>
  struct tuple_size;

  template <typename T1, typename T2>
  struct tuple_size<pair<T1, T2>> {
    static constexpr size_t value = 2;
  };

  template <typename... Ts>
  struct tuple_size<tuple<Ts...>> {
    static constexpr size_t value = sizeof...(Ts);
  };

  template <typename T, size_t N>
  struct tuple_size<T[N]> {
    static constexpr size_t value = N;
  };

  template <size_t I, typename T1, typename T2>
  typename tuple_element<I, pair<T1, T2>>::type &
  get(pair<T1, T2> &p) {
    if constexpr (I == 0) return p.first;
    else return p.second;
  }

  template <size_t I, typename T1, typename T2>
  typename tuple_element<I, pair<T1, T2>>::type &&
  get(pair<T1, T2> &&p) {
    if constexpr (I == 0) return static_cast<T1&&>(p.first);
    else return static_cast<T2&&>(p.second);
  }

  template <size_t I, typename T, typename... Ts>
  auto& get(tuple<T, Ts...> &t) {
    if constexpr (I == 0) return t.head;
    else return get<I-1>(t.tail);
  }

  template <size_t I, typename T, typename... Ts>
  auto&& get(tuple<T, Ts...> &&t) {
    if constexpr (I == 0) return static_cast<T&&>(t.head);
    else return get<I-1>(static_cast<tuple<Ts...>&&>(t.tail));
  }

  // array
  template <typename T, size_t N>
  struct array {
    T data[N];
    T& operator[](size_t i) { return data[i]; }
    const T& operator[](size_t i) const { return data[i]; }
  };

  template <size_t I, typename T, size_t N>
  struct tuple_element<I, array<T, N>> { using type = T; };

  template <typename T, size_t N>
  struct tuple_size<array<T, N>> {
    static constexpr size_t value = N;
  };

  template <size_t I, typename T, size_t N>
  T& get(array<T, N> &a) { return a.data[I]; }

  template <size_t I, typename T, size_t N>
  T&& get(array<T, N> &&a) { return static_cast<T&&>(a.data[I]); }
}

void use(int);

void test_pair() {
  auto [a, b] = std::make_pair(1, 2);
  // expected-note@-1{{'a' declared here}}
  // expected-note@-2{{'b' declared here}}
#pragma omp parallel
  {
    use(a + b);
    // expected-error@-1{{capturing tuple-like structured binding 'a' is not yet supported in OpenMP}}
    // expected-error@-2{{capturing tuple-like structured binding 'b' is not yet supported in OpenMP}}
  }
}

void test_tuple() {
  std::tuple<int, int, int> t = {1, 2, 3};
  auto [x, y, z] = t;
  // expected-note@-1{{'x' declared here}}
  // expected-note@-2{{'y' declared here}}
  // expected-note@-3{{'z' declared here}}
#pragma omp parallel
  {
    use(x + y + z);
    // expected-error@-1{{capturing tuple-like structured binding 'x' is not yet supported in OpenMP}}
    // expected-error@-2{{capturing tuple-like structured binding 'y' is not yet supported in OpenMP}}
    // expected-error@-3{{capturing tuple-like structured binding 'z' is not yet supported in OpenMP}}
  }
}

void test_array() {
  std::array<int, 2> arr = {1, 2};
  auto [p, q] = arr;
  // expected-note@-1{{'p' declared here}}
  // expected-note@-2{{'q' declared here}}
#pragma omp parallel
  {
    use(p + q);
    // expected-error@-1{{capturing tuple-like structured binding 'p' is not yet supported in OpenMP}}
    // expected-error@-2{{capturing tuple-like structured binding 'q' is not yet supported in OpenMP}}
  }
}
