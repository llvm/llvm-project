// RUN: %clang_cc1 -std=c++2a -verify %s

namespace p3 {
  void bar(...);
  template <typename... Args> void foo(Args... args) {
    (void)[... xs = args] {
      bar(xs...);
    };
  }

  void use() {
    foo();
    foo(1);
  }
}

template<typename ...T> void f(T ...t) {
  (void)[&...x = t] {
    x; // expected-error {{unexpanded parameter pack 'x'}}
  };

  // Not OK: can't expand 'x' outside its scope.
  weird((void)[&...x = t] {
    return &x; // expected-error {{unexpanded parameter pack 'x'}}
  }...         // expected-error {{does not contain any unexpanded}}
  );

  // OK, capture only one 'slice' of 'x'.
  weird((void)[&x = t] {
    return &x;
  }...
  );

  // 'x' is not expanded by the outer '...', but 'T' is.
  weird((void)[&... x = t] {
    return T() + &x; // expected-error {{unexpanded parameter pack 'x'}}
  }...               // expected-error {{does not contain any unexpanded}}
  );
}

template<int ...a> constexpr auto x = [...z = a] (auto F) { return F(z...); };
static_assert(x<1,2,3>([](int a, int b, int c) { return 100 * a + 10 * b + c; }) == 123);
static_assert(x<1,2,3>([](int a, int b, int c) { return 100 * a + 10 * b + c; }) == 124); // expected-error {{failed}} \
                                                                                          // expected-note {{evaluates to '123 == 124'}}

template<int ...a> constexpr auto y = [z = a...] (auto F) { return F(z...); }; // expected-error {{must appear before the name of the capture}}
static_assert(y<1,2,3>([](int a, int b, int c) { return 100 * a + 10 * b + c; }) == 123);
static_assert(y<1,2,3>([](int a, int b, int c) { return 100 * a + 10 * b + c; }) == 124); // expected-error {{failed}} \
                                                                                          // expected-note {{evaluates to '123 == 124'}}
