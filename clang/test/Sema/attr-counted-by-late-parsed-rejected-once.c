// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fexperimental-late-parse-attributes -fsyntax-only -verify %s

// Late-parsed counted_by-family attributes report the same errors as eager
// ones. A rejected declaration-specifier attribute is reported once, not once
// per declarator that shares it, and each attribute on a declarator is checked
// even after an earlier one is rejected.

#define __counted_by(f) __attribute__((counted_by(f)))
#define __sized_by(f) __attribute__((sized_by(f)))

struct shared_decl_spec {
  int n;
  // expected-error@+1{{'counted_by' only applies to pointers or C99 flexible array members}}
  int __counted_by(n) a, b, c;
};

struct two_rejected_attrs {
  int n;
  int m;
  // expected-error@+2{{'counted_by' only applies to pointers or C99 flexible array members}}
  // expected-error@+1{{'sized_by' only applies to pointers}}
  int __counted_by(n) __sized_by(m) x;
};

struct two_rejected_shared {
  int n;
  int m;
  // expected-error@+2{{'counted_by' only applies to pointers or C99 flexible array members}}
  // expected-error@+1{{'sized_by' only applies to pointers}}
  int __counted_by(n) __sized_by(m) a, b;
};
