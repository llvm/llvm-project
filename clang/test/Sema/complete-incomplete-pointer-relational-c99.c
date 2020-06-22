// RUN: %clang_cc1 -fsyntax-only -verify -std=c99 -Wc11-extensions %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c89 -Wc11-extensions %s

int incomplete[]; // expected-warning {{tentative array definition assumed to have one element}}
int complete[6];

int test_comparison_between_incomplete_and_complete_pointer() {
  return (&incomplete < &complete) &&  // expected-warning {{pointer comparisons before C11 need to be between two complete or two incomplete types; 'int (*)[]' is incomplete and 'int (*)[6]' is complete}}
         (&incomplete <= &complete) && // expected-warning {{pointer comparisons before C11 need to be between two complete or two incomplete types; 'int (*)[]' is incomplete and 'int (*)[6]' is complete}}
         (&incomplete > &complete) &&  // expected-warning {{pointer comparisons before C11 need to be between two complete or two incomplete types; 'int (*)[]' is incomplete and 'int (*)[6]' is complete}}
         (&incomplete >= &complete) && // expected-warning {{pointer comparisons before C11 need to be between two complete or two incomplete types; 'int (*)[]' is incomplete and 'int (*)[6]' is complete}}
         (&incomplete == &complete) &&
         (&incomplete != &complete);
}
