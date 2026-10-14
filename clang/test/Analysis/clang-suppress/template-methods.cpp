// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify %s

void clang_analyzer_warnIfReached();

// Systematic tests for [[clang::suppress]] on template methods inside
// non-template and template classes.

// Placeholder types for triggering instantiations.
// - Type{A,B} should match an unconstrained template type parameter.
struct TypeA{};
struct TypeB{};

// ============================================================================
// Group A: Non-template class with suppressed/unsuppressed template methods
// ============================================================================

struct NonTemplateClassWithTemplatedMethod {
  template <typename T>
  [[clang::suppress]] void suppressed(T) {
    clang_analyzer_warnIfReached(); // no-warning
  }

  template <typename T>
  void unsuppressed(T) {
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  }
};

void test_nontpl_class() {
  NonTemplateClassWithTemplatedMethod().suppressed(TypeA{});
  NonTemplateClassWithTemplatedMethod().unsuppressed(TypeA{});
}

// ============================================================================
// Group B: Template class with template methods — inline
// ============================================================================

template <typename T>
struct TemplateClassWithTemplateInlineMethod {
  template <typename U>
  [[clang::suppress]] void suppressed(U) {
    clang_analyzer_warnIfReached(); // no-warning
  }

  template <typename U>
  void unsuppressed(U) {
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  }
};

void test_tpl_class_tpl_inline_method() {
  TemplateClassWithTemplateInlineMethod<TypeA>().suppressed(TypeB{});
  TemplateClassWithTemplateInlineMethod<TypeA>().unsuppressed(TypeB{});
}

// ============================================================================
// Group C: Template class with template methods — out-of-line
// ============================================================================

template <typename T>
struct TemplateClassWithTemplateOOLMethod {
  template <typename U>
  [[clang::suppress]] void suppress_at_decl_outline(U);

  template <typename U>
  void suppress_at_def_outline(U);
};

// Attribute on declaration only — NOT honored at out-of-line definition.
template <typename T>
template <typename U>
void TemplateClassWithTemplateOOLMethod<T>::suppress_at_decl_outline(U) {
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

// Attribute on out-of-line definition — suppressed.
template <typename T>
template <typename U>
[[clang::suppress]] void TemplateClassWithTemplateOOLMethod<T>::suppress_at_def_outline(U) {
  clang_analyzer_warnIfReached(); // no-warning
}

void test_tpl_class_tpl_ool_method() {
  TemplateClassWithTemplateOOLMethod<TypeA>().suppress_at_decl_outline(TypeB{});
  TemplateClassWithTemplateOOLMethod<TypeA>().suppress_at_def_outline(TypeB{});
}

// ============================================================================
// Group D: Template-template parameters
// ============================================================================

// A simple "box" template used as a template-template argument.
template <typename T>
struct Box {
  void get() {
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  }
};

// A version of Box that suppresses its own methods.
template <typename T>
class [[clang::suppress]] SuppressedBox {
public:
  void get() {
    clang_analyzer_warnIfReached(); // no-warning
  }
};

// Adaptor whose own methods are suppressed; the contained Box's methods are not.
template <typename T, template <typename> class Container>
class [[clang::suppress]] SuppressedAdaptor {
public:
  Container<T> data;

  void adaptor_method() {
    clang_analyzer_warnIfReached(); // no-warning
  }
};

// Adaptor with no suppression; Box's own suppression is independent.
template <typename T, template <typename> class Container>
struct UnsuppressedAdaptor {
  Container<T> data;

  void adaptor_method() {
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  }
};

void test_template_template() {
  // SuppressedAdaptor<Box>: adaptor method suppressed; Box::get not affected.
  SuppressedAdaptor<TypeA, Box>().adaptor_method();  // suppressed by adaptor's attr
  SuppressedAdaptor<TypeA, Box>().data.get();        // warns — Box has no attr, different lexical context

  // UnsuppressedAdaptor<SuppressedBox>: adaptor warns; SuppressedBox::get suppressed.
  UnsuppressedAdaptor<TypeA, SuppressedBox>().adaptor_method();  // warns — adaptor has no attr
  UnsuppressedAdaptor<TypeA, SuppressedBox>().data.get();        // suppressed by SuppressedBox's attr
}
