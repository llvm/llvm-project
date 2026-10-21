// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -fconstexpr-depth=100 -verify=expected,both %s
// RUN: %clang_cc1                                         -fconstexpr-depth=100 -verify=ref,both      %s

constexpr int f(int a) {
  if (a == 100)
    return 1 / 0; // both-warning {{division by zero is undefined}}

  return f(a + 1); // both-note {{exceeded maximum depth of 100 calls}} \
                   // both-note {{in call to 'f(99)'}} \
                   // both-note {{in call to 'f(98)'}} \
                   // both-note {{in call to 'f(97)'}} \
                   // both-note {{in call to 'f(96)'}} \
                   // both-note {{in call to 'f(95)'}} \
                   // both-note {{skipping 90 calls in backtrace}} \
                   // both-note {{in call to 'f(4)'}} \
                   // both-note {{in call to 'f(3)'}} \
                   // both-note {{in call to 'f(2)'}} \
                   // both-note {{in call to 'f(1)'}}
}
static_assert(f(0) == 100, ""); // both-error {{not an integral constant expression}} \
                                // both-note {{in call to 'f(0)'}}
