// RUN: %clang_cc1 -std=c++20 -fsyntax-only -Wno-unused-value -Wno-stack-exhausted -fconstexpr-depth=1024 %s

int rand();

constexpr int a(int) {
  {
    (100000000001024 ^ a(0) * 0 ? 2147483647 : rand()) ? 2147483647 : 1;
  }
  return 0;
}
