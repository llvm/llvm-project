// RUN: %clang_analyze_cc1 -analyzer-checker core,nullability -w -verify %s

// expected-no-diagnostics

id _Nonnull conjure_nonnull(void);
void use_nullable(_Nullable id x);

id _Nonnull foo(void) {
  void *j = conjure_nonnull();
  use_nullable(j);
  return j; // no-warning
}
