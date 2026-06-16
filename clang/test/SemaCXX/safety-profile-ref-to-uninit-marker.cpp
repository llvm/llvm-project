// RUN: %clang_cc1 -fsyntax-only -verify=expected -fprofiles -std=c++23 %s
// RUN: %clang_cc1 -fsyntax-only -verify=no-profiles -std=c++23 %s

// The [[ref_to_uninit]] marker is recognized regardless of -fprofiles and,
// with no profile enforced, has no effect on valid pointer/reference subjects.

// expected-no-diagnostics
// no-profiles-no-diagnostics

int g;

int *gp [[ref_to_uninit]] = &g;
int &gr [[ref_to_uninit]] = g;

[[ref_to_uninit]] int *gp_prefix = &g;

[[ref_to_uninit]] int *allocate(int n);
void fill(int *p [[ref_to_uninit]]);
void bind(int &r [[ref_to_uninit]]);

struct S {
  int *m [[ref_to_uninit]];
};
