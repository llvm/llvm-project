// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify %s

void clang_analyzer_warnIfReached();

// Tests for [[clang::suppress]] with diagnostic identifier arguments.

// ============================================================================
// Group A: Bare [[clang::suppress]] vs. with identifier
// ============================================================================

void bare_suppress() {
  [[clang::suppress]] {
    clang_analyzer_warnIfReached(); // no-warning: bare suppress works
  }
}

void suppress_with_identifier() {
  // FIXME: This should suppress debug.ExprInspection warnings, but currently
  // any identifier makes the suppression a no-op.
  [[clang::suppress("debug.ExprInspection")]] {
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  }
}

void suppress_with_wrong_identifier() {
  // Even with the wrong checker name, the current behavior is the same:
  // any identifier makes the suppression a no-op.
  [[clang::suppress("alpha.SomeOtherChecker")]] {
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  }
}

// ============================================================================
// Group B: Identifier on declarations
// ============================================================================

[[clang::suppress]] void decl_bare_suppress() {
  clang_analyzer_warnIfReached(); // no-warning
}

// FIXME: Should suppress, but currently identifiers disable suppression.
[[clang::suppress("debug.ExprInspection")]] void decl_with_identifier() {
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

// ============================================================================
// Group C: Identifier on class
// ============================================================================

struct [[clang::suppress]] C_BareSuppressedClass {
  void method() {
    clang_analyzer_warnIfReached(); // no-warning
  }
};

// FIXME: Should suppress, but identifiers disable suppression.
struct [[clang::suppress("core")]] C_IdentifierSuppressedClass {
  void method() {
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  }
};

// ============================================================================
// Group D: Multiple identifiers
// ============================================================================

void multiple_identifiers() {
  // FIXME: Multiple identifiers — currently treated as a no-op.
  [[clang::suppress("core.NullDereference", "core.DivideZero")]] {
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  }
}

// ============================================================================
// Group E: Empty string identifier
// ============================================================================

void empty_string_identifier() {
  // An empty string is still a non-empty identifier list.
  [[clang::suppress("")]] {
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  }
}

// ============================================================================
// Group F: Mixed — bare suppress and identifier suppress in same function
// ============================================================================

void mixed_suppressions() {
  [[clang::suppress]] {
    clang_analyzer_warnIfReached(); // no-warning: bare suppress works
  }

  // FIXME: Should suppress too, but identifiers disable it.
  [[clang::suppress("debug.ExprInspection")]] {
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  }
}

// ============================================================================
// Group G: Identifier on namespace
// ============================================================================

namespace [[clang::suppress]] G_BareNS {
  void func() {
    clang_analyzer_warnIfReached(); // no-warning
  }
} // namespace G_BareNS

// FIXME: Should suppress, but identifiers disable it.
namespace [[clang::suppress("core")]] G_IdentifierNS {
  void func() {
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  }
} // namespace G_IdentifierNS
