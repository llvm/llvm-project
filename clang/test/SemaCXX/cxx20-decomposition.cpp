// RUN: %clang_cc1 -fsyntax-only -std=c++20 -verify %s
// expected-no-diagnostics

template <typename, typename>
constexpr bool is_same = false;
template <typename T>
constexpr bool is_same<T, T> = true;

struct S {
  int i;
  int &j;
};

void check_category() {
  int a = 42;
  {
    auto [v, r] = S{1, a};
    (void)[ v, r ] {
      static_assert(is_same<decltype(v), int>);
      static_assert(is_same<decltype(r), int &>);
    };
  }
  {
    auto [v, r] = S{1, a};
    (void)[&v, &r ] {
      static_assert(is_same<decltype(v), int>);
      static_assert(is_same<decltype(r), int &>);
    };
  }
  {
    S s{1, a};
    const auto &[v, r] = s;
    (void)[ v, r ] {
      static_assert(is_same<decltype(v), const int>);
      static_assert(is_same<decltype(r), int &>);
    };
  }
  {
    S s{1, a};
    const auto &[v, r] = s;
    (void)[&v, &r ] {
      static_assert(is_same<decltype(v), const int>);
      static_assert(is_same<decltype(r), int &>);
    };
  }
}

void check_array() {
  int arr[2] = {42, 42};
  auto &[a, b] = arr;
  (void)[ a, &b ] {
    static_assert(is_same<decltype(a), int>);
    static_assert(is_same<decltype(b), int>);
  };
}

struct tuple {
  template <unsigned long I>
  decltype(auto) get() {
    if constexpr (I == 0) {
      return a;
    } else {
      return b;
    }
  }

  template <unsigned long I>
  decltype(auto) get() const {
    if constexpr (I == 0) {
      return a;
    } else {
      return b;
    }
  }

  int a = 0;
  int &b = a;
};

namespace std {

template <typename T>
struct tuple_size {
  static constexpr unsigned long value = 2;
};

template <unsigned long, typename T>
struct tuple_element;

template <>
struct tuple_element<0, tuple> {
  using type = int;
};

template <>
struct tuple_element<1, tuple> {
  using type = int &;
};

template <>
struct tuple_element<0, const tuple> {
  using type = int;
};

template <>
struct tuple_element<1, const tuple> {
  using type = const int &;
};
} // namespace std

void check_tuple_like() {
  tuple t;
  {
    auto [v, r] = t;
    (void)[ v, r ] {
      static_assert(is_same<decltype(v), int>);
      static_assert(is_same<decltype(r), int &>);
    };
  }
  {
    auto &[v, r] = t;
    (void)[&v, &r ] {
      static_assert(is_same<decltype(v), int>);
      static_assert(is_same<decltype(r), int &>);
    };
  }
  {
    const auto &[v, r] = t;
    (void)[ v, r ] {
      static_assert(is_same<decltype(v), int>);
      static_assert(is_same<decltype(r), const int &>);
    };
  }
  {
    const auto &[v, r] = t;
    (void)[&v, &r ] {
      static_assert(is_same<decltype(v), int>);
      static_assert(is_same<decltype(r), const int &>);
    };
  }
}
