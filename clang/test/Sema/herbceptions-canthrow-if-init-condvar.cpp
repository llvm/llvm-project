// RUN: %clang_cc1 -std=c++26 -fherbceptions -fsyntax-only %s
// expected-no-diagnostics

namespace std {
inline constexpr bool is_constant_evaluated() noexcept { return __builtin_is_constant_evaluated(); }
}

struct BoolWrapper {
  bool value;
  constexpr operator bool() const { return value; }
};

// Reproducer for a Sema::canThrow null deref on an IfStmt with a C++17
// init-statement + condition variable, e.g. `if (T x = init;)`.
// IfStmt::getCond() is null for this shape. Before the fix,
// Sema::canThrow called canThrow(IS->getCond()) without a null check and
// segfaulted inside Stmt::getStmtClassName when S was null.
//
// The crash was reached by a 'throws' template function whose body
// contained an 'if' with a condition variable being instantiated; the
// instantiation triggered canThrow on the body via
// ActOnFinishFunctionBody. The non-template case here suffices because
// parsing a 'throws' function that contains a bare call to a throws
// function inside a non-throws lambda walks the lambda body through
// Sema::canThrow.

inline constexpr void throwing() throws {}

inline constexpr void outer() throws {
  auto lam = [] {
    if (BoolWrapper b{true}; b) {
      throwing();
    }
  };
  lam();
}

int main() { outer(); }
