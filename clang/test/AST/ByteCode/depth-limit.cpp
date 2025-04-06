// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -fconstexpr-depth=100 -verify %s
// RUN: %clang_cc1 -fconstexpr-depth=100 -verify=ref %s

constexpr int f(int a) {
  if (a == 100)
    return 1 / 0; // expected-warning {{division by zero is undefined}} \
                  // ref-warning {{division by zero is undefined}}

  return f(a + 1); // ref-note {{exceeded maximum depth of 100 calls}} \
                   // ref-note {{in call to 'f(99)'}} \
                   // ref-note {{in call to 'f(98)'}} \
                   // ref-note {{in call to 'f(97)'}} \
                   // ref-note {{in call to 'f(96)'}} \
                   // ref-note {{in call to 'f(95)'}} \
                   // ref-note {{skipping 90 calls in backtrace}} \
                   // ref-note {{in call to 'f(4)'}} \
                   // ref-note {{in call to 'f(3)'}} \
                   // ref-note {{in call to 'f(2)'}} \
                   // ref-note {{in call to 'f(1)'}} \
                   // expected-note {{exceeded maximum depth of 100 calls}} \
                   // expected-note {{in call to 'f(99)'}} \
                   // expected-note {{in call to 'f(98)'}} \
                   // expected-note {{in call to 'f(97)'}} \
                   // expected-note {{in call to 'f(96)'}} \
                   // expected-note {{in call to 'f(95)'}} \
                   // expected-note {{skipping 90 calls in backtrace}} \
                   // expected-note {{in call to 'f(4)'}} \
                   // expected-note {{in call to 'f(3)'}} \
                   // expected-note {{in call to 'f(2)'}} \
                   // expected-note {{in call to 'f(1)'}}
}
static_assert(f(0) == 100, ""); // ref-error {{not an integral constant expression}} \
                                // ref-note {{in call to 'f(0)'}} \
                                // expected-error {{not an integral constant expression}} \
                                // expected-note {{in call to 'f(0)'}}
