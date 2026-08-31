// RUN: %clang_cc1 -fsyntax-only -std=c++2c -ferror-limit 1 -verify %s

// GH214917: don't crash expanding an expansion statement after a fatal error.

unknown_type a; // expected-error {{unknown type name 'unknown_type'}}
unknown_type b;
// expected-error@* {{too many errors emitted}}

struct V {
  int i, j;
};

void bar() {
  template for (auto i : V{42, 10}) {
    (void)i;
  }
}
