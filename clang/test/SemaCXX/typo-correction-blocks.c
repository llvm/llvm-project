// RUN: %clang_cc1 -triple i386-apple-macosx -fblocks -fsyntax-only -verify %s

extern int h(int *);
extern void g(int, void (^)(void));
extern int fuzzys;                  // expected-note {{'fuzzys' declared here}}

static void f(void *v) {
  g(fuzzy, ^{                       // expected-error {{did you mean 'fuzzys'}}
    int i = h(v);
  });
}

