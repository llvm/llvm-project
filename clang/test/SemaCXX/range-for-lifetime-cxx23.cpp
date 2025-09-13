// RUN: %clang_cc1 -std=c++23 -fsyntax-only -verify %s

using size_t = decltype(sizeof(void *));

namespace std {
template <typename T> struct vector {
  T &operator[](size_t I);
};

struct string {
  const char *begin();
  const char *end();
};

} // namespace std

std::vector<std::string> getData();

void foo() {
  // Verifies we don't trigger a diagnostic from -Wdangling-gsl
  // when iterating over a temporary in C++23.
  for (auto c : getData()[0]) {
    (void)c;
  }
}

// expected-no-diagnostics
