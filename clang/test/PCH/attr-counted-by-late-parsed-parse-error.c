// RUN: %clang_cc1 -fexperimental-late-parse-attributes -fallow-pcm-with-compiler-errors -emit-pch -o %t %s -verify=header
// RUN: %clang_cc1 -fexperimental-late-parse-attributes -fallow-pcm-with-compiler-errors -include-pch %t -fsyntax-only -verify %s

// A late-parsed counted_by whose argument fails to parse leaves no attribute
// to complete the CountAttributedType built for it. The node used to keep a
// null count, which reached the AST and asserted when the PCH was read back
// (`use getIncompleteCountAttributedType for a null count`). The node is now
// completed with a RecoveryExpr and the field is marked invalid, like an
// argument that parses but is rejected.

#ifndef HEADER
#define HEADER

#define __counted_by(f) __attribute__((counted_by(f)))

struct empty_arg {
  int n;
  int *__counted_by() p; // header-error {{argument required after attribute}}
};

struct bad_expr {
  int n;
  int *__counted_by(n +) p; // header-error {{expected expression}}
};

// A declaration-specifier attribute shares one node between declarators.
typedef int *IP;
struct shared_node {
  int n;
  IP __counted_by(n +) a, b; // header-error {{expected expression}}
};

#else

// expected-no-diagnostics

int use(struct empty_arg *e, struct bad_expr *b, struct shared_node *s) {
  return e->p[0] + b->p[0] + s->a[0] + s->b[0];
}

#endif
