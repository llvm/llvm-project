// RUN: %check_clang_tidy -std=c++20 %s bugprone-assignment-in-if-condition %t

void testRequires() {
  if constexpr (requires(int &a) { a = 0; }) {
  }
}
