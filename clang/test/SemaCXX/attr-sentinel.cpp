// RUN: %clang_cc1 -fsyntax-only -verify=expected,cxx17 -std=c++17 %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected,cxx23 -std=c++23 %s

void f(int, ...) __attribute__((sentinel));

void g() {
  f(1, 2, __null);
}

typedef __typeof__(sizeof(int)) size_t;

struct S {
  S(int,...) __attribute__((sentinel)); // expected-note {{marked sentinel}}
  void a(int,...) __attribute__((sentinel)); // expected-note {{marked sentinel}}
  void* operator new(size_t,...) __attribute__((sentinel)); // expected-note {{marked sentinel}}
  void operator()(int,...) __attribute__((sentinel)); // expected-note {{marked sentinel}}
};

void class_test() {
  S s(1,2,3); // expected-warning {{missing sentinel in function call}}
  S* s2 = new (1,2,3) S(1, __null); // expected-warning {{missing sentinel in function call}}
  s2->a(1,2,3); // expected-warning {{missing sentinel in function call}}
  s(1,2,3); // expected-warning {{missing sentinel in function call}}
}

namespace consistent_diganostic_test {
struct Ty {
    void baseline0(int, ...) __attribute__((sentinel));     // #BASELINE0_NOTE
    void baseline1(int, ...) __attribute__((sentinel(1)));  // #BASELINE1_NOTE
    void baseline2(int, ...) __attribute__((sentinel(2)));  // #BASELINE2_NOTE

    template<typename T>
    int&& foo(T&& val, ...) __attribute__((sentinel(1))); // #FOO_NOTE

    template<class Self>
    int&& bar(this Self&& self, ...) __attribute__((sentinel(1))); // #BAR_NOTE
    // cxx17-error@-1 {{explicit object parameters are incompatible with C++ standards before C++2b}}
};

void test_baseline0() {
    auto sty = Ty{};

    sty.baseline0(1);
    // expected-warning@-1 {{not enough variable arguments in 'baseline0' declaration to fit a sentinel}}
    // expected-note@#BASELINE0_NOTE {{function has been explicitly marked sentinel here}}

    sty.baseline0(1, 2);
    // expected-warning@-1 {{missing sentinel in function call}}
    // expected-note@#BASELINE0_NOTE {{function has been explicitly marked sentinel here}}

    sty.baseline0(1, 2, nullptr);
}

void test_baseline1() {
    auto sty = Ty{};

    sty.baseline1(1, 3);
    // expected-warning@-1 {{not enough variable arguments in 'baseline1' declaration to fit a sentinel}}
    // expected-note@#BASELINE1_NOTE {{function has been explicitly marked sentinel here}}

    sty.baseline1(1, 2, 3);
    // expected-warning@-1 {{missing sentinel in function call}}
    // expected-note@#BASELINE1_NOTE {{function has been explicitly marked sentinel here}}

    sty.baseline1(1, 2, nullptr, 3);
}

void test_baseline2() {
    auto sty = Ty{};

    sty.baseline2(1, 2, 3);
    // expected-warning@-1 {{not enough variable arguments in 'baseline2' declaration to fit a sentinel}}
    // expected-note@#BASELINE2_NOTE {{function has been explicitly marked sentinel here}}

    sty.baseline2(1, 2, 3, 4);
    // expected-warning@-1 {{missing sentinel in function call}}
    // expected-note@#BASELINE2_NOTE {{function has been explicitly marked sentinel here}}

    sty.baseline2(1, 2, nullptr, 3, 4);
}

void test_foo() {
    auto sty = Ty{};

    sty.foo(1, 2);
    // expected-warning@-1 {{not enough variable arguments in 'foo' declaration to fit a sentinel}}
    // expected-note@#FOO_NOTE {{function has been explicitly marked sentinel here}}

    sty.foo(1, 2, 3);
    // expected-warning@-1 {{missing sentinel in function call}}
    // expected-note@#FOO_NOTE {{function has been explicitly marked sentinel here}}

    sty.foo(1, nullptr, 3);
}

void test_bar() {
    auto sty = Ty{};

    sty.bar(1, 2);
    // expected-warning@-1 {{not enough variable arguments in 'bar' declaration to fit a sentinel}}
    // expected-note@#BAR_NOTE {{function has been explicitly marked sentinel here}}

    sty.bar(1, 2, 3);
    // expected-warning@-1 {{missing sentinel in function call}}
    // expected-note@#BAR_NOTE {{function has been explicitly marked sentinel here}}

    sty.bar(1, nullptr, 3);
}
} // namespace consistent_diganostic_test
