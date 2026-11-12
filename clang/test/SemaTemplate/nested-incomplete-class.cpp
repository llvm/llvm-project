// RUN: %clang_cc1 -fsyntax-only %s

template <typename T>
struct foo {
  struct bar;

  bar fn() {
    // Should not get errors about bar being incomplete here.
    bar b = bar(1, 2);
    return b;
  }
};

template <typename T>
struct foo<T>::bar {
  bar(int, int);
};

void fn() {
  foo<int>().fn();
}
