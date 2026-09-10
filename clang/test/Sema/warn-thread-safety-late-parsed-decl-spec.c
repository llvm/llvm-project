// RUN: %clang_cc1 -fsyntax-only -Wthread-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wthread-safety -fexperimental-late-parse-attributes -verify %s

// In declaration-specifier position of a struct member, only counted_by-family
// type attributes are late-parsed. Declaration attributes such as guarded_by
// are parsed there eagerly, as without late parsing, so they apply to every
// declarator through the declaration specifier. They used to be late-parsed
// and attached to the first declarator only.

#define __counted_by(f) __attribute__((counted_by(f)))

struct __attribute__((capability("mutex"))) mutex {
  int x;
};

typedef int *IP;

struct decl_spec_guarded_by {
  struct mutex mu;
  __attribute__((guarded_by(mu))) int a, b;
  __attribute__((pt_guarded_by(mu))) int *p, *q;
};

void write_guarded(struct decl_spec_guarded_by *s) {
  s->a = 1; // expected-warning {{writing variable 'a' requires holding mutex 's->mu' exclusively}}
  s->b = 1; // expected-warning {{writing variable 'b' requires holding mutex 's->mu' exclusively}}
  *s->p = 1; // expected-warning {{writing the value pointed to by 'p' requires holding mutex 's->mu' exclusively}}
  *s->q = 1; // expected-warning {{writing the value pointed to by 'q' requires holding mutex 's->mu' exclusively}}
}

// A late-parsed type attribute and an eager declaration attribute in the same
// declaration specifier.
struct decl_spec_counted_by_and_guarded_by {
  struct mutex mu;
  int n;
  IP __counted_by(n) __attribute__((guarded_by(mu))) a, b;
};

void write_counted_guarded(struct decl_spec_counted_by_and_guarded_by *s) {
  s->a = 0; // expected-warning {{writing variable 'a' requires holding mutex 's->mu' exclusively}}
  s->b = 0; // expected-warning {{writing variable 'b' requires holding mutex 's->mu' exclusively}}
}

// Parsed eagerly, so the mutex must be declared first.
struct decl_spec_guarded_by_mutex_after {
  __attribute__((guarded_by(mu))) int a, b; // expected-error {{use of undeclared identifier 'mu'}}
  struct mutex mu;
};
