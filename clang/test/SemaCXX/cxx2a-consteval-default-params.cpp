// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++23 %s

consteval int undefined();  // expected-note 2 {{declared here}}

void check_lambdas_unused(
    int a = [](int no_error = undefined()) {
        return no_error;
    }(0),
    int b = [](int defaulted = undefined()) {
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
    }(),  // expected-note {{in the default initializer of 'defaulted'}}
    int d = [](int defaulted = sizeof(undefined())) {
        return defaulted;
    }()
) {
    return 0;
}

int test_check_lambdas_used = check_lambdas_used();

struct UnusedInitWithLambda {
    int a = [] {
        return undefined(); // never evaluated because immediate escalating
    }();
    // UnusedInitWithLambda is never constructed, so the initializer
    // of b and undefined() are never evaluated.
    int b = [](int no_error = undefined()) {
        return no_error;
    }();
};

consteval int ub(int n) {
    return 0/n;
}

struct InitWithLambda { // expected-note {{'InitWithLambda' is an immediate constructor because the default initializer of 'b' contains a call to a consteval function 'undefined' and that call is not a constant expression}}
    int b = [](int error = undefined()) {  // expected-note {{undefined function 'undefined' cannot be used in a constant expression}}
        return error;
    }();
    int c = [](int error = sizeof(undefined()) + ub(0)) {

        return error;
    }();
} i;
// expected-error@-1 {{call to immediate function 'InitWithLambda::InitWithLambda' is not a constant expression}} \
   expected-note@-1 {{in call to 'InitWithLambda()'}}

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
