// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fexperimental-late-parse-attributes -fsyntax-only -verify %s

// A pointer can have only one counted_by-family attribute. A second one used
// to stack another CountAttributedType on the first, with or without late
// parsing.

#define __counted_by(f) __attribute__((counted_by(f)))
#define __counted_by_or_null(f) __attribute__((counted_by_or_null(f)))
#define __sized_by(f) __attribute__((sized_by(f)))
#define __sized_by_or_null(f) __attribute__((sized_by_or_null(f)))

typedef int *IP;

struct same_attr {
  int n;
  int m;
  // expected-error@+1{{'counted_by' attribute cannot be applied to a pointer that already has a 'counted_by' attribute}}
  int *__counted_by(n) __counted_by(m) p;
};

struct same_count {
  int n;
  // expected-error@+1{{'counted_by' attribute cannot be applied to a pointer that already has a 'counted_by' attribute}}
  int *__counted_by(n) __counted_by(n) p;
};

struct different_attrs {
  int n;
  int m;
  // expected-error@+1{{'sized_by' attribute cannot be applied to a pointer that already has a 'counted_by' attribute}}
  int *__counted_by(n) __sized_by(m) p;
  // expected-error@+1{{'counted_by_or_null' attribute cannot be applied to a pointer that already has a 'sized_by_or_null' attribute}}
  int *__sized_by_or_null(n) __counted_by_or_null(m) q;
};

// One in declaration-specifier position, shared by both declarators, is
// diagnosed once.
struct shared_decl_spec {
  int n;
  int m;
  // expected-error@+1{{'sized_by' attribute cannot be applied to a pointer that already has a 'counted_by' attribute}}
  IP __counted_by(n) __sized_by(m) a, b;
};

// One in declaration-specifier position and one in the declarator.
struct decl_spec_and_declarator {
  int n;
  int m;
  // expected-error@+1{{'sized_by' attribute cannot be applied to a pointer that already has a 'counted_by' attribute}}
  IP __counted_by(n) a __sized_by(m);
};

// One in the pointer chunk and one after the declarator.
struct chunk_and_after_declarator {
  int n;
  int m;
  // expected-error@+1{{'sized_by' attribute cannot be applied to a pointer that already has a 'counted_by' attribute}}
  int *__counted_by(n) p __sized_by(m);
};
