// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify %s

void clang_analyzer_warnIfReached();

// Systematic tests for [[clang::suppress]] interaction with macros.
//
// The fullyContains() function compares source ranges using
// SourceManager::isBeforeInTranslationUnit, which handles macro
// expansion locations. These tests verify that suppression works
// correctly when bugs are reported inside macro expansions.

// Placeholder type for triggering instantiations.
struct Type{};

// ============================================================================
// Group A: Bug inside macro, suppression outside (using warnIfReached)
// ============================================================================

#define WARN clang_analyzer_warnIfReached()

void macro_in_suppressed_block() {
  [[clang::suppress]] {
    WARN; // no-warning
  }
}

void macro_in_unsuppressed_block() {
  WARN; // expected-warning{{REACHABLE}}
}

// ============================================================================
// Group B: Function-like macro with expression
// ============================================================================

#define DO_WARN() clang_analyzer_warnIfReached()

void funclike_macro_suppressed() {
  [[clang::suppress]] {
    DO_WARN(); // no-warning
  }
}

void funclike_macro_unsuppressed() {
  DO_WARN(); // expected-warning{{REACHABLE}}
}

// ============================================================================
// Group C: Nested macros
// ============================================================================

#define INNER_WARN() clang_analyzer_warnIfReached()
#define OUTER_WARN() INNER_WARN()

void nested_macro_suppressed() {
  [[clang::suppress]] {
    OUTER_WARN(); // no-warning
  }
}

void nested_macro_unsuppressed() {
  OUTER_WARN(); // expected-warning{{REACHABLE}}
}

// ============================================================================
// Group D: Macro defining entire function body
// ============================================================================

#define BUGGY_BODY { clang_analyzer_warnIfReached(); }

[[clang::suppress]] void func_with_macro_body()
  BUGGY_BODY // no-warning

void func_with_macro_body_unsuppressed()
  BUGGY_BODY // expected-warning{{REACHABLE}}

// ============================================================================
// Group E: Macro in suppressed class method
// ============================================================================

struct [[clang::suppress]] MacroInSuppressedClass {
  void method() {
    WARN; // no-warning
  }
};

struct MacroInUnsuppressedClass {
  void method() {
    WARN; // expected-warning{{REACHABLE}}
  }
};

void test_E() {
  MacroInSuppressedClass().method();
  MacroInUnsuppressedClass().method();
}

// ============================================================================
// Group F: Macro expanding to suppression attribute + code
// ============================================================================

#define SUPPRESS_AND_WARN [[clang::suppress]] clang_analyzer_warnIfReached()

void macro_suppression_wrapper() {
  SUPPRESS_AND_WARN; // no-warning
}

// ============================================================================
// Group G: Macro in template context
// ============================================================================

template <typename T>
struct [[clang::suppress]] MacroInTemplate {
  void method() {
    WARN; // no-warning
  }
};

template <typename T>
struct MacroInTemplate_NoAttr {
  void method() {
    WARN; // expected-warning{{REACHABLE}}
  }
};

void test_G() {
  MacroInTemplate<Type>().method();
  MacroInTemplate_NoAttr<Type>().method();
}

// ============================================================================
// Group H: Null dereference through direct null, suppressed at statement level
// ============================================================================

int macro_deref_suppressed() {
  int *p = 0;
  [[clang::suppress]] return *p; // no-warning
}

int macro_deref_unsuppressed() {
  int *p = 0;
  return *p; // expected-warning{{Dereference of null pointer (loaded from variable 'p')}}
}

// ============================================================================
// Group I: Stringification and token pasting (shouldn't affect suppression)
// ============================================================================

#define STRINGIFY(x) #x
#define CONCAT(a, b) a##b

void stringify_suppressed() {
  [[clang::suppress]] {
    const char *s = STRINGIFY(hello);
    (void)s;
    int CONCAT(var, 1) = 0;
    clang_analyzer_warnIfReached(); // no-warning
    (void)var1;
  }
}

void stringify_unsuppressed() {
  const char *s = STRINGIFY(hello);
  (void)s;
  int CONCAT(var, 1) = 0;
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  (void)var1;
}

// ============================================================================
// Group J: Multi-line macro with warnIfReached
// ============================================================================

#define MULTI_LINE_WARN  \
  do {                   \
    clang_analyzer_warnIfReached(); \
  } while (0)

void multiline_macro_suppressed() {
  [[clang::suppress]] {
    MULTI_LINE_WARN; // no-warning
  }
}

void multiline_macro_unsuppressed() {
  MULTI_LINE_WARN; // expected-warning{{REACHABLE}}
}
