// RUN: %clang_cc1 %s -std=c++20 -fsyntax-only -verify
// expected-no-diagnostics

// Reported by: https://github.com/llvm/llvm-project/issues/57013
// The following code should not crash clang
struct X {
  char arr[2];
  constexpr X() {}
  constexpr void modify() {
    arr[0] = 0;
  }
};
constexpr X f(X t) {
    t.modify();
    return t;
}
auto x = f(X());
