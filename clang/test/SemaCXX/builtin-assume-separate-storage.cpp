// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=ref,both %s
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify=expected,both -fexperimental-new-constant-interpreter %s

// __builtin_assume_separate_storage is not constant-evaluable (#225335).

constexpr int separated(int *a, int *b) { // ref-error {{constexpr function never produces a constant expression}}
  __builtin_assume_separate_storage(a, b); // ref-note 2 {{subexpression not valid in a constant expression}} \
                                           // expected-note {{subexpression not valid in a constant expression}}
  return 0;
}

constexpr int a = 1, b = 2;
static_assert(separated(const_cast<int *>(&a), const_cast<int *>(&b)) == 0); // both-error {{static assertion expression is not an integral constant expression}} \
                                                                              // both-note {{in call to 'separated(&a, &b)'}}

// Original reproducer from #225335
static_assert([] { // both-error {{static assertion expression is not an integral constant expression}} \
                   // both-note {{in call to '[] {}.operator()()'}}
  int i, j;
  __builtin_assume_separate_storage(&i, &j); // both-note {{subexpression not valid in a constant expression}}
  return true;
}());
