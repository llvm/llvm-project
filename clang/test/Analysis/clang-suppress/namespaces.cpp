// RUN: %clang_analyze_cc1 -analyzer-checker=core -verify %s

// Systematic tests for [[clang::suppress]] on namespaces.

// ============================================================================
// Group A: Attributed namespace suppresses inline definitions
// ============================================================================

namespace [[clang::suppress]]
suppressed_namespace {
  int foo() {
    int *x = 0;
    return *x; // no-warning: inside attributed namespace
  }

  int ool_foo();
}

// Out-of-line definition in an attributed namespace is NOT suppressed.
int suppressed_namespace::ool_foo() {
    int *x = 0;
    return *x; // expected-warning{{Dereference of null pointer (loaded from variable 'x')}}
}

// ============================================================================
// Group B: Reopened namespace (without attribute) is NOT suppressed
// ============================================================================

// Another instance of the same namespace — the attribute does not carry over.
namespace suppressed_namespace {
  int bar() {
    int *x = 0;
    return *x; // expected-warning{{Dereference of null pointer (loaded from variable 'x')}}
  }
}
