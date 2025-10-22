// RUN: %clang_cc1 -x c++ -fsyntax-only -verify %s

struct fam_struct {
  int x;
  char count;
  int array[] __attribute__((counted_by(count))); // expected-warning {{'counted_by' attribute ignored}}
};

