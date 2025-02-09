// RUN: %clang_cc1 -DNEEDS_LATE_PARSING -fno-experimental-late-parse-attributes -fsyntax-only -verify %s
// RUN: %clang_cc1 -DNEEDS_LATE_PARSING -fsyntax-only -verify %s

// RUN: %clang_cc1 -UNEEDS_LATE_PARSING -fno-experimental-late-parse-attributes -fsyntax-only -verify=ok %s
// RUN: %clang_cc1 -UNEEDS_LATE_PARSING -fsyntax-only -verify=ok %s

#define __counted_by(f)  __attribute__((counted_by(f)))

struct size_known { int dummy; };

#ifdef NEEDS_LATE_PARSING
struct on_decl {
  // expected-error@+1{{use of undeclared identifier 'count'}}
  struct size_known *buf __counted_by(count);
  int count;
};

#else

// ok-no-diagnostics
struct on_decl {
  int count;
  struct size_known *buf __counted_by(count);
};

#endif
