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
