// RUN: %clang_cc1 -DNEEDS_LATE_PARSING -fno-experimental-late-parse-attributes -fsyntax-only -verify %s
// RUN: %clang_cc1 -DNEEDS_LATE_PARSING -fsyntax-only -verify %s

// RUN: %clang_cc1 -UNEEDS_LATE_PARSING -fno-experimental-late-parse-attributes -fsyntax-only -verify=ok %s
// RUN: %clang_cc1 -UNEEDS_LATE_PARSING -fsyntax-only -verify=ok %s

#define __sized_by_or_null(f)  __attribute__((sized_by_or_null(f)))

struct size_known { int dummy; };

#ifdef NEEDS_LATE_PARSING
struct on_decl {
  // expected-error@+1{{use of undeclared identifier 'count'}}
  struct size_known *buf __sized_by_or_null(count);
  int count;
};

#else

// ok-no-diagnostics
struct on_decl {
  int count;
  struct size_known *buf __sized_by_or_null(count);
};

#endif
