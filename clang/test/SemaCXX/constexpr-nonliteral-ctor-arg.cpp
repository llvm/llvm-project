// RUN: %clang_cc1 -std=c++17 -fsyntax-only -verify=pre23 %s
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=pre23 %s
// RUN: %clang_cc1 -std=c++23 -fsyntax-only -verify=cxx23 %s
// cxx23-no-diagnostics

struct NonLiteral {
  NonLiteral() {}
};

struct Literal {
  constexpr Literal() {}
};

struct Swallow {
  template <typename T>
  constexpr Swallow(T);
};

// Must be diagnosed before C++23.
constexpr int bad() { Swallow s(NonLiteral{}); return 42; } // pre23-error {{constexpr function never produces a constant expression}} \
                                                            // pre23-note {{non-literal type 'NonLiteral' cannot be used in a constant expression}}

// Must NOT be diagnosed: literal-type and scalar arguments.
constexpr int ok1() { Swallow s(Literal{}); return 42; }
constexpr int ok2() { Swallow s(1); return 42; }

template <typename T>
constexpr Swallow::Swallow(T) {}
