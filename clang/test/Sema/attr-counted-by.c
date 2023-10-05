// RUN: %clang_cc1 -fstrict-flex-arrays=3 -fsyntax-only -verify %s

#define __counted_by(f)  __attribute__((counted_by(f)))

struct bar;

struct not_found {
  int count;
  struct bar *fam[] __counted_by(bork); // expected-error {{field 'bork' in 'counted_by' not found}}
};

struct not_found_suggest {
  int bork; // expected-note {{'bork' declared here}}
  struct bar *fam[] __counted_by(blork); // expected-error {{field 'blork' in 'counted_by' not found; did you mean 'bork'?}}
};

int global; // expected-note {{variable 'global' is declared here}}

struct found_outside_of_struct {
  int bork;
  struct bar *fam[] __counted_by(global); // expected-error {{field 'global' in 'counted_by' is not found in struct}}
};

struct self_referrential {
  int bork;
  struct bar *self[] __counted_by(self); // expected-error {{field 'self' in 'counted_by' cannot refer to the flexible array}}
};

struct non_int {
  double non_integer; // expected-error {{field 'non_integer' in 'counted_by' is not a non-boolean integer type}}
  struct bar *fam[] __counted_by(non_integer); // expected-note {{field 'non_integer' declared here}}
};

struct array_of_ints {
  int non_integer[2]; // expected-error {{field 'non_integer' in 'counted_by' is not a non-boolean integer type}}
  struct bar *fam[] __counted_by(non_integer); // expected-note {{field 'non_integer' declared here}}
};

struct not_a_fam {
  double non_integer;
  struct bar *non_fam __counted_by(non_integer); // expected-error {{'counted_by' only applies to flexible array members}}
};
