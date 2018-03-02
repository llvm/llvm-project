// RUN: %check_clang_tidy %s misc-misplaced-const %t -- -- -std=c++17

// This test previously would cause a failed assertion because the structured
// binding declaration had no valid type associated with it. This ensures the
// expected clang diagnostic is generated instead.
// CHECK-MESSAGES: :[[@LINE+1]]:6: error: decomposition declaration '[x]' requires an initializer [clang-diagnostic-error]
auto [x];

struct S { int a; };
S f();

int main() {
  auto [x] = f();
}

