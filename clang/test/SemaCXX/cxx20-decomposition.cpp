// RUN: %clang_cc1 -fsyntax-only -std=c++20 -verify -Wunused-variable %s

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
struct tuple_size;

template <typename T>
struct tuple_size<T&> : tuple_size<T>{};

template <typename T>
requires requires { tuple_size<T>::value; }
struct tuple_size<const T> : tuple_size<T>{};

template <>
struct tuple_size<tuple> {
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

namespace ODRUseTests {
  struct P { int a; int b; };
  void GH57826() {
    const auto [a, b] = P{1, 2}; //expected-note 2{{'b' declared here}} \
                                 //expected-note 3{{'a' declared here}}
    (void)[&](auto c) { return b + [&a] {
        return a;
    }(); }(0);
    (void)[&](auto c) { return b + [&a](auto) {
        return a;
    }(0); }(0);
    (void)[=](auto c) { return b + [&a](auto) {
        return a;
    }(0); }(0);
    (void)[&a,&b](auto c) { return b + [&a](auto) {
        return a;
    }(0); }(0);
    (void)[&a,&b](auto c) { return b + [a](auto) {
        return a;
    }(0); }(0);
    (void)[&a](auto c) { return b + [&a](auto) { // expected-error 2{{variable 'b' cannot be implicitly captured}} \
                                                 // expected-note 2{{lambda expression begins here}} \
                                                 // expected-note 4{{capture 'b'}}
        return a;
    }(0); }(0); // expected-note {{in instantiation}}
    (void)[&b](auto c) { return b + [](auto) {   // expected-note 3{{lambda expression begins here}} \
                                                 // expected-note 6{{capture 'a'}} \
                                                 // expected-note 6{{default capture}} \
                                                 // expected-note {{in instantiation}} \
                                                 // expected-note {{while substituting into a lambda}}
        return a;  // expected-error 3{{variable 'a' cannot be implicitly captured}}
    }(0); }(0); // expected-note 2{{in instantiation}}
  }
}


namespace GH95081 {
    void prevent_assignment_check() {
        int arr[] = {1,2};
        auto [e1, e2] = arr;

        auto lambda = [e1] {
            e1 = 42;  // expected-error {{cannot assign to a variable captured by copy in a non-mutable lambda}}
        };
    }

    void f(int&) = delete;
    void f(const int&);

    int arr[1];
    void foo() {
        auto [x] = arr;
        [x]() {
            f(x); // deleted f(int&) used to be picked up erroneously
        } ();
    }
}
