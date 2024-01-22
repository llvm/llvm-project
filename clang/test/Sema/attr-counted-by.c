// RUN: %clang_cc1 -fsyntax-only -verify %s

#define __counted_by(f)  __attribute__((counted_by(f)))

struct bar;

struct not_found {
  int count;
  struct bar *fam[] __counted_by(bork); // expected-error {{use of undeclared identifier 'bork'}}
};

struct no_found_count_not_in_substruct {
  unsigned long flags;
  unsigned char count; // expected-note {{field 'count' declared here}}
  struct A {
    int dummy;
    int array[] __counted_by(count); // expected-error {{'counted_by' field 'count' isn't within the same struct as the flexible array}}
  } a;
};

struct not_found_suggest {
  int bork; // expected-note {{'bork' declared here}}
  struct bar *fam[] __counted_by(blork); // expected-error {{use of undeclared identifier 'blork'; did you mean 'bork'?}}
};

int global; // expected-note {{'global' declared here}}

struct found_outside_of_struct {
  int bork;
  struct bar *fam[] __counted_by(global); // expected-error {{field 'global' in 'counted_by' not inside structure}}
};

struct self_referrential {
  int bork;
  struct bar *self[] __counted_by(self); // expected-error {{'counted_by' cannot refer to the flexible array 'self'}}
};

struct non_int_count {
  double dbl_count; // expected-note {{field 'dbl_count' declared here}}
  struct bar *fam[] __counted_by(dbl_count); // expected-error {{field 'dbl_count' in 'counted_by' must be a non-boolean integer type}}
};

struct array_of_ints_count {
  int integers[2]; // expected-note {{field 'integers' declared here}}
  struct bar *fam[] __counted_by(integers); // expected-error {{field 'integers' in 'counted_by' must be a non-boolean integer type}}
};

struct not_a_fam {
  int count;
  struct bar *non_fam __counted_by(count); // expected-error {{'counted_by' only applies to C99 flexible array members}}
};

struct not_a_c99_fam {
  int count;
  struct bar *non_c99_fam[0] __counted_by(count); // expected-error {{'counted_by' only applies to C99 flexible array members}}
};

struct annotated_with_anon_struct {
  unsigned long flags;
  struct {
    unsigned char count; // expected-note {{'count' declared here}}
    int array[] __counted_by(crount); // expected-error {{use of undeclared identifier 'crount'; did you mean 'count'?}}
  };
};
