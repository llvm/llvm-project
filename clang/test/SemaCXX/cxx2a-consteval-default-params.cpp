// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++23 %s

consteval int undefined();  // expected-note 4 {{declared here}}

void check_lambdas_unused(
    int a = []
    {
        // The body of a lambda is not a subexpression of the lambda
        // so this is immediately evaluated even if the parameter
        // is never used.
        return undefined();  // expected-error {{not a constant expression}} \
                             // expected-note  {{undefined function 'undefined'}}
    }(),
    int b = [](int no_error = undefined()) {
        return no_error;
    }(0),
    int c = [](int defaulted = undefined()) {
        return defaulted;
    }()
) {}

int check_lambdas_used(
    int b = [](int no_error = undefined()) {
        return no_error;
    }(0),
    int c = [](int defaulted = undefined()) { // expected-error {{not a constant expression}} \
                              // expected-note  {{declared here}} \
                              // expected-note  {{undefined function 'undefined'}}
        return defaulted;
    }(),  // expected-note {{in the default initalizer of 'defaulted'}}
    int d = [](int defaulted = sizeof(undefined())) {
        return defaulted;
    }()
) {
    return 0;
}

int test_check_lambdas_used = check_lambdas_used();

struct UnusedInitWithLambda {
    int a = [] {
        return undefined();  // expected-error {{not a constant expression}} \
                             // expected-note  {{undefined function 'undefined'}}
    }();
    // UnusedInitWithLambda is never constructed, so the initializer
    // of b and undefined() are never evaluated.
    int b = [](int no_error = undefined()) {
        return no_error;
    }();
};

consteval int ub(int n) {
    return 0/n; // expected-note  {{division}}
}

struct InitWithLambda {
    int b = [](int error = undefined()) { // expected-error {{not a constant expression}} \
                              // expected-note  {{declared here}} \
                              // expected-note  {{undefined function 'undefined'}}
        return error;
    }(); // expected-note {{in the default initalizer of 'error'}}
    int c = [](int error = sizeof(undefined()) + ub(0)) { // expected-error {{'ub' is not a constant expression}} \
                                                          // expected-note  {{declared here}} \
                                                          // expected-note {{in call to 'ub(0)}}
        return error;
    }(); // expected-note {{in the default initalizer of 'error'}}
} i; // expected-note {{in implicit default constructor}}

namespace ShouldNotCrash {
    template<typename T>
    struct F {
        template<typename U>
        F(const U&) {}
    };
    struct A {
        static constexpr auto x = [] {};
        F<int> f = x;
    };
    void f(A a = A()) { }
}

namespace GH62224 {
  consteval int fwd();
  template <int i = fwd()>
  struct C {
    consteval C(int = fwd()) { }
    consteval int get() { return i; }
  };

  consteval int fwd() { return 42; }
  C<> Val; // No error since fwd is defined already.
  static_assert(Val.get() == 42);
}
