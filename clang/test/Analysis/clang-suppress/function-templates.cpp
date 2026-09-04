// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify %s

void clang_analyzer_warnIfReached();

// Systematic tests for [[clang::suppress]] on function templates and their
// explicit specializations.

// Placeholder types for triggering instantiations.
// - Type should match an unconstrained template type parameter.
// - Specialized should match a specialization pattern.
struct Type{};
struct Specialized{};

// ============================================================================
// Group A: Attribute on forward declaration only — NOT honored at definition
// ============================================================================

template <typename T> [[clang::suppress]] void FunctionTemplateSuppressed(T);
template <typename T>
void FunctionTemplateSuppressed(T) {
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

template <typename T>
void FunctionTemplateUnsuppressed(T) {
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

void test_fwd_decl_attr() {
  FunctionTemplateSuppressed(Type{});
  FunctionTemplateUnsuppressed(Type{});
}

// ============================================================================
// Group B: Explicit full function specialization — attribute on specialization
// ============================================================================

// Only the Specialized specialization is suppressed.
template <typename T>
void ExplicitSpecAttrOnSpec(T) {
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

template <>
[[clang::suppress]] void ExplicitSpecAttrOnSpec(Specialized) {
  clang_analyzer_warnIfReached(); // no-warning
}

void test_attr_on_spec() {
  ExplicitSpecAttrOnSpec(Type{});                // warns (primary)
  ExplicitSpecAttrOnSpec(Specialized{});         // suppressed (explicit specialization)
}

// ============================================================================
// Group C: Explicit full function specialization — attribute on primary
// ============================================================================

// Only the primary template is suppressed.
template <typename T>
[[clang::suppress]] void ExplicitSpecAttrOnPrimary(T) {
  clang_analyzer_warnIfReached(); // no-warning
}

template <>
void ExplicitSpecAttrOnPrimary(Specialized) {
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

void test_attr_on_primary() {
  ExplicitSpecAttrOnPrimary(Type{});             // suppressed (primary)
  ExplicitSpecAttrOnPrimary(Specialized{});      // warns (explicit specialization)
}

// ============================================================================
// Group D: Variadic template with suppress + explicit specialization override
// ============================================================================

template <typename... Args>
[[clang::suppress]] void Variadic_Suppressed(Args...) {
  clang_analyzer_warnIfReached(); // no-warning
}

// Variadic template function specialization — NOT suppressed.
template <>
void Variadic_Suppressed(Type, Specialized) {
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

void test_variadic() {
  Variadic_Suppressed();
  Variadic_Suppressed(Type{});
  Variadic_Suppressed(Type{}, Specialized{});
}
