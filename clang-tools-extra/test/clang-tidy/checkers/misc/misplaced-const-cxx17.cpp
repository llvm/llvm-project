// RUN: %check_clang_tidy -std=c++17-or-later -expect-clang-tidy-error %s misc-misplaced-const %t

// This test previously would cause a failed assertion because the structured
// binding declaration had no valid type associated with it. This ensures the
// expected clang diagnostic is generated instead.
// CHECK-MESSAGES: :[[@LINE+1]]:6: error: structured binding declaration '[x]' requires an initializer [clang-diagnostic-error]
auto [x];

struct S { int a; };
S f();

int main() {
  auto [x] = f();
}

