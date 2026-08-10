// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyzer-inline-max-stack-depth=5 -w -std=c++17 -verify %s

void clang_analyzer_eval(bool);

namespace inline_large_functions_with_if_constexpr {
bool f0() { if constexpr (true); return true; }
bool f1() { if constexpr (true); return f0(); }
bool f2() { if constexpr (true); return f1(); }
bool f3() { if constexpr (true); return f2(); }
bool f4() { if constexpr (true); return f3(); }
bool f5() { if constexpr (true); return f4(); }
bool f6() { if constexpr (true); return f5(); }
bool f7() { if constexpr (true); return f6(); }
void bar() {
  clang_analyzer_eval(f7()); // expected-warning{{TRUE}}
}
} // namespace inline_large_functions_with_if_constexpr
