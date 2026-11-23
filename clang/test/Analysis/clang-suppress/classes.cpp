// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify %s

void clang_analyzer_warnIfReached();

// Systematic tests for [[clang::suppress]] on non-template classes and methods.

// ============================================================================
// Group A: Attribute on class — inline method suppressed, out-of-line not
// ============================================================================

// Placeholder type for triggering instantiations.
struct Type{};

class [[clang::suppress]] SuppressedClass {
  void foo() {
    clang_analyzer_warnIfReached(); // no-warning: inline method in suppressed class
  }

  void bar();
};

void SuppressedClass::bar() {
  clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
}

// ============================================================================
// Group B: Attribute on method declaration vs definition
// ============================================================================

class SuppressedMethodClass {
  // Attribute on the inline definition — suppressed.
  [[clang::suppress]] void foo() {
    clang_analyzer_warnIfReached(); // no-warning
  }

  // Attribute on the in-class declaration only — NOT honored at out-of-line def.
  [[clang::suppress]] void bar1();

  // No attribute on the in-class declaration.
  void bar2();
};

void SuppressedMethodClass::bar1() {
  clang_analyzer_warnIfReached(); // expected-warning {{REACHABLE}}
}

// Attribute on the out-of-line definition — suppressed.
[[clang::suppress]]
void SuppressedMethodClass::bar2() {
  clang_analyzer_warnIfReached(); // no-warning
}

// ============================================================================
// Group C: Template member function with early instantiation
// ============================================================================

// The suppression mechanism walks the lexical DeclContext chain to find
// suppression attributes. This test verifies that the walk follows template
// instantiation patterns (not just primary templates) when the instantiation
// point precedes the definition.

struct Clazz {
  template <typename T>
  static void templated_memfn();
};

// This must come before the 'templated_memfn' is defined!
void instantiate() {
  Clazz::templated_memfn<Type>();
}

template <typename T>
void Clazz::templated_memfn() {
  [[clang::suppress]] clang_analyzer_warnIfReached(); // no-warning
}
