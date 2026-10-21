// RUN: %clang_cc1 %s -std=c++14 -fsyntax-only -verify
// RUN: %clang_cc1 %s -std=c++20 -fsyntax-only -verify
// RUN: %clang_cc1 %s -std=c++2c -fsyntax-only -verify

// RUN: %clang_cc1 %s -std=c++14 -fsyntax-only -fexperimental-new-constant-interpreter -verify
// RUN: %clang_cc1 %s -std=c++20 -fsyntax-only -fexperimental-new-constant-interpreter -verify
// RUN: %clang_cc1 %s -std=c++2c -fsyntax-only -fexperimental-new-constant-interpreter -verify

template <typename T>
constexpr T foo(T a);   // expected-note {{declared here}}

int main() {
  int k = foo<int>(5);  // Ok
  constexpr int j =     // expected-error {{constexpr variable 'j' must be initialized by a constant expression}}
          foo<int>(5);  // expected-note {{undefined function 'foo<int>' cannot be used in a constant expression}}
}

template <typename T>
constexpr T foo(T a) {
  return a;
}

#if __cplusplus > 202002L

namespace GH115118 {

struct foo {
    // expected-note@-1 2{{while}}
    foo(const foo&) = default;
    foo(auto)
        requires([]<int = 0>() -> bool { return true; }())
        // expected-error@-1 {{non-constant expression}}
        // expected-note@-2 {{undefined function}} \
        // expected-note@-2 {{declared}}
    {}
};

// FIXME: This will be fixed by https://github.com/llvm/llvm-project/pull/205557
struct bar {
    // expected-note@-1 {{while}}
    foo x; // check that the lambda gets instantiated.
};

}  // namespace GH115118

#endif
