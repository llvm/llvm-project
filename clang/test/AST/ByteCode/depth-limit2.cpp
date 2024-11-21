// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -fconstexpr-depth=2 -verify %s
// RUN: %clang_cc1 -fconstexpr-depth=2 -verify=ref %s


constexpr int func() {
  return 12;
}

constexpr int foo() {
  return func(); // expected-note {{exceeded maximum depth of 2 calls}} \
                 // ref-note {{exceeded maximum depth of 2 calls}}
}

constexpr int bar() {
  return foo(); // expected-note {{in call to 'foo()'}} \
                // ref-note {{in call to 'foo()'}}
}

static_assert(bar() == 12, ""); // expected-error {{not an integral constant expression}} \
                                // expected-note {{in call to 'bar()'}} \
                                // ref-error {{not an integral constant expression}} \
                                // ref-note {{in call to 'bar()'}}

