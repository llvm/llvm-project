// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

// __builtin_assume_separate_storage is not constant-evaluable (#225335).

constexpr int separated(int *a, int *b) { // expected-error {{constexpr function never produces a constant expression}}
  __builtin_assume_separate_storage(a, b); // expected-note 2 {{subexpression not valid in a constant expression}}
  return 0;
}

constexpr int a = 1, b = 2;
static_assert(separated(const_cast<int *>(&a), const_cast<int *>(&b)) == 0); // expected-error {{static assertion expression is not an integral constant expression}} \
                                                                              // expected-note {{in call to 'separated(&a, &b)'}}
