// RUN: %clang_cc1 %s -verify -fsyntax-only

struct S {
  Unknown u; // expected-error {{unknown type name 'Unknown'}}
  int i;
};
// Should not crash
struct S s[] = {[0].i = 0, [1].i = 1, {}};
