// RUN: %clang_cc1 %s -verify -Wconversion

#define P(X) _Pragma(#X)
#define V(X) X

#define X \
  P(clang diagnostic push) \
  P(clang diagnostic ignored "-Wconversion") \
  ) = 1.2; \
  P(clang diagnostic pop)

void f() {
  int a = 1.2; // expected-warning {{changes value}}

  // Note, we intentionally enter a tentatively-parsed context here to trigger
  // regular use of lookahead. This would go wrong if _Pragma checking in macro
  // argument pre-expansion also tries to use token lookahead.
  int (b
  V(X)

  int c = 1.2; // expected-warning {{changes value}}
}
