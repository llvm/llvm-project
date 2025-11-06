// Fixes #166328
// RUN: %clang_cc1 -std=c++14 -fsyntax-only -verify= %s

// This function requires a non-type template argument, but we'll
// intentionally misuse it by omitting the argument.
// This should trigger an error, but must not crash the compiler when
// used in a compile-time evaluated switch case statement.
template<int n>
constexpr int nonTypeTemplateParamFunc() { // expected-note 5 {{candidate template ignored: couldn't infer template argument 'n'}}
  return n;
}

constexpr int switchWithErrs(int n) {
  switch (n) {
  case int(nonTypeTemplateParamFunc()): // expected-error {{no matching function for call to 'nonTypeTemplateParamFunc'}}
    return n;
  // GNU Case ranges (Extension)
  case 0 ... int(nonTypeTemplateParamFunc()): // expected-error {{no matching function for call to 'nonTypeTemplateParamFunc'}}
    return n;
  case nonTypeTemplateParamFunc() ... 100: // expected-error {{no matching function for call to 'nonTypeTemplateParamFunc'}}
    return n + 1;
  case nonTypeTemplateParamFunc() ... nonTypeTemplateParamFunc(): // expected-error 2 {{no matching function for call to 'nonTypeTemplateParamFunc'}}
    return n + 2;
  default:
    return 0;
  }
}

auto main() -> int {
  constexpr auto num = switchWithErrs(1); // expected-error {{constexpr variable 'num' must be initialized by a constant expression}}
}
