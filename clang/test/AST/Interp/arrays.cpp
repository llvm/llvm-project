// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -verify %s
// RUN: %clang_cc1 -verify=ref %s


/// expected-no-diagnostics
/// ref-no-diagnostics

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc99-extensions"
#pragma clang diagnostic ignored "-Winitializer-overrides"
/// FIXME: The example below tests ImplicitValueInitExprs, but we can't
///   currently evaluate other parts of it.
#if 0
struct fred {
  char s [6];
  int n;
};

struct fred y [] = { [0] = { .s[0] = 'q' } };
#endif
#pragma clang diagnostic pop
