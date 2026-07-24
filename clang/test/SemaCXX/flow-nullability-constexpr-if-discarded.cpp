// Regression test: warn_null_init_nonnull ("null assigned to a variable of
// nonnull type") must NOT fire in a discarded `if constexpr` branch. The
// discarded branch is parsed inside a DiscardedStatement evaluation context,
// so the type-based null-init check in SemaDecl is gated on that.

// RUN: %clang_cc1 -fsyntax-only -fflow-sensitive-nullability -std=c++17 -verify %s

// Non-template: discarded branch is silent, live branch warns.
void plain_discarded() {
  if constexpr (false) {
    int *_Nonnull p = nullptr; // no warning: discarded branch
    (void)p;
  }
}

void plain_live() {
  if constexpr (true) {
    int *_Nonnull p = nullptr; // expected-warning{{null assigned to a variable of nonnull type}}
    (void)p;
  }
}

// Template with a non-dependent (literal) constexpr condition: the discarded
// arm is parsed in a DiscardedStatement context, so the warning is suppressed
// at the template's initial parse and the discarded body is not re-instantiated
// — no warning, even across multiple instantiations. (A live arm is avoided
// here: warn_null_init_nonnull in a live arm fires once per instantiation,
// which is orthogonal to this discarded-branch fix.)
template <class T>
void tmpl_discarded() {
  if constexpr (false) {
    int *_Nonnull q = nullptr; // no warning: discarded branch
    (void)q;
  }
}

void instantiate() {
  tmpl_discarded<int>();
  tmpl_discarded<char>();
}
