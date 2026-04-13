// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify %s

void clang_analyzer_warnIfReached();

// ============================================================================
// Group A: Basic class template — attribute on primary
// ============================================================================

// Placeholder types for triggering instantiations.
// - Type{A,B} should match an unconstrained template type parameter.
// - Specialized{A,B} should match some specialization pattern.
struct TypeA{};
struct TypeB{};
struct SpecializedA{};
struct SpecializedB{};

template <typename T>
class [[clang::suppress]] A_Primary {
public:
  void inline_method() {
    clang_analyzer_warnIfReached(); // no-warning
  }
  void outline_method();
  static void static_inline() {
    clang_analyzer_warnIfReached(); // no-warning
  }
  static void static_outline();
};

template <typename T>
void A_Primary<T>::outline_method() {
  // Out-of-line: lexical context is the translation unit.
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

template <typename T>
void A_Primary<T>::static_outline() {
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

void test_A() {
  A_Primary<TypeA>().inline_method();
  A_Primary<TypeA>().outline_method();
  A_Primary<TypeA>::static_inline();
  A_Primary<TypeA>::static_outline();
  // Different instantiation.
  A_Primary<TypeB>().inline_method();
}

// ============================================================================
// Group B: Explicit full specialization — attribute isolation
// ============================================================================

// --- B1: attribute on primary only ---

template <typename T>
class [[clang::suppress]] B1_AttrOnPrimary {
public:
  void method() {
    clang_analyzer_warnIfReached(); // no-warning
  }
};

// Explicit specialization is independent — NOT suppressed.
template <>
struct B1_AttrOnPrimary<SpecializedA> {
  void method() {
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  }
};

void test_B1() {
  B1_AttrOnPrimary<TypeA>().method();           // suppressed (primary)
  B1_AttrOnPrimary<SpecializedA>().method();    // warns (spec, no attr)
}

// --- B2: attribute on specialization only ---

template <typename T>
struct B2_AttrOnSpec {
  void method() {
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  }
};

template <>
class [[clang::suppress]] B2_AttrOnSpec<SpecializedA> {
public:
  void method() {
    clang_analyzer_warnIfReached(); // no-warning
  }
};

void test_B2() {
  B2_AttrOnSpec<TypeA>().method();              // warns (primary, no attr)
  B2_AttrOnSpec<SpecializedA>().method();       // suppressed (spec has attr)
}

// --- B3: attribute on both primary and specialization ---

template <typename T>
class [[clang::suppress]] B3_AttrOnBoth {
public:
  void method() {
    clang_analyzer_warnIfReached(); // no-warning
  }
};

template <>
class [[clang::suppress]] B3_AttrOnBoth<SpecializedA> {
public:
  void method() {
    clang_analyzer_warnIfReached(); // no-warning
  }
};

void test_B3() {
  B3_AttrOnBoth<TypeA>().method();              // suppressed
  B3_AttrOnBoth<SpecializedA>().method();       // suppressed
}

// ============================================================================
// Group C: Partial specializations
// ============================================================================

// --- C1: attribute on partial specialization only ---

template <typename T, typename U>
struct C1_AttrOnPartial {
  void method() {
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  }
};

template <typename T>
class [[clang::suppress]] C1_AttrOnPartial<T, SpecializedA> {
public:
  void method() {
    clang_analyzer_warnIfReached(); // no-warning
  }
};

void test_C1() {
  C1_AttrOnPartial<TypeA, TypeA>().method();            // warns (primary, no attr)
  C1_AttrOnPartial<TypeA, SpecializedA>().method();     // suppressed (partial spec)
}

// --- C2: attribute on primary, partial spec has none ---

template <typename T, typename U>
class [[clang::suppress]] C2_AttrOnPrimary {
public:
  void method() {
    clang_analyzer_warnIfReached(); // no-warning
  }
};

template <typename T>
struct C2_AttrOnPrimary<T, SpecializedA> {
  void method() {
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  }
};

void test_C2() {
  C2_AttrOnPrimary<TypeA, TypeA>().method();            // suppressed (primary)
  C2_AttrOnPrimary<TypeA, SpecializedA>().method();     // warns (partial spec, no attr)
}

// --- C3: two partial specializations, only one suppressed ---

template <typename T, typename U>
struct C3_TwoPartials {
  void method() {
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  }
};

template <typename T>
class [[clang::suppress]] C3_TwoPartials<T, SpecializedA> {
public:
  void method() {
    clang_analyzer_warnIfReached(); // no-warning
  }
};

template <typename T>
struct C3_TwoPartials<T, SpecializedB> {
  void method() {
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  }
};

void test_C3() {
  C3_TwoPartials<TypeA, TypeA>().method();              // warns (primary)
  C3_TwoPartials<TypeA, SpecializedA>().method();       // suppressed (first partial)
  C3_TwoPartials<TypeA, SpecializedB>().method();       // warns (second partial, no attr)
}

// ============================================================================
// Group D: Forward-declared class template (chooseDefinitionRedecl path)
// ============================================================================

// The template is forward-declared, then defined. chooseDefinitionRedecl()
// must find the definition among the redeclarations.

// --- D1: Forward-declared without attribute, defined with attribute ---
template <typename T>
class D1_ForwardDeclared;

template <typename T>
class [[clang::suppress]] D1_ForwardDeclared {
public:
  void method() {
    clang_analyzer_warnIfReached(); // no-warning
  }
};

void test_D1() {
  D1_ForwardDeclared<TypeA>().method();
}

// --- D2: Forward-declared without attribute, defined without attribute ---
template <typename T>
struct D2_ForwardDeclared_NoAttr;

template <typename T>
struct D2_ForwardDeclared_NoAttr {
  void method() {
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  }
};

void test_D2() {
  D2_ForwardDeclared_NoAttr<TypeA>().method();
}

// ============================================================================
// Group E: Specialization with out-of-line (OOL) methods
// ============================================================================

template <typename T>
class [[clang::suppress]] E_SpecWithOOL {
public:
  void inline_method() {
    clang_analyzer_warnIfReached(); // no-warning
  }
  void outline_method();
};

template <typename T>
void E_SpecWithOOL<T>::outline_method() {
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

// Explicit specialization with attribute and out-of-line method.
template <>
class [[clang::suppress]] E_SpecWithOOL<SpecializedA> {
public:
  void inline_method() {
    clang_analyzer_warnIfReached(); // no-warning
  }
  void outline_method();
};

// Out-of-line for the specialization — not suppressed.
void E_SpecWithOOL<SpecializedA>::outline_method() {
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

void test_E() {
  E_SpecWithOOL<TypeA>().inline_method();
  E_SpecWithOOL<TypeA>().outline_method();
  E_SpecWithOOL<SpecializedA>().inline_method();
  E_SpecWithOOL<SpecializedA>().outline_method();
}

// ============================================================================
// Group F: Nested class inside class template specialization
// ============================================================================

template <typename T>
class [[clang::suppress]] F_Outer {
public:
  struct Inner {
    void method() {
      clang_analyzer_warnIfReached(); // no-warning
    }
  };
};

template <typename T>
struct F_Outer_NoAttr {
  struct Inner {
    void method() {
      clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
    }
  };
};

void test_F() {
  F_Outer<TypeA>::Inner().method();
  F_Outer_NoAttr<TypeA>::Inner().method();
}

// ============================================================================
// Group G: Class template with default template arguments
// ============================================================================

template <typename T, typename U = TypeA>
class [[clang::suppress]] G_WithDefault {
public:
  void method() {
    clang_analyzer_warnIfReached(); // no-warning
  }
};

template <typename T, typename U = TypeA>
struct G_WithDefault_NoAttr {
  void method() {
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  }
};

void test_G() {
  G_WithDefault<TypeA>().method();               // uses default U=TypeA
  G_WithDefault<TypeA, TypeB>().method();        // explicit U=TypeB
  G_WithDefault_NoAttr<TypeA>().method();        // uses default U=TypeA
}

// ============================================================================
// Group H: Explicit instantiation directive
// ============================================================================

template <typename T>
class [[clang::suppress]] H_ExplicitInst {
public:
  void method() {
    clang_analyzer_warnIfReached(); // no-warning
  }
};

// Explicit instantiation.
template class H_ExplicitInst<SpecializedA>;

void test_H() {
  H_ExplicitInst<SpecializedA>().method();
}
