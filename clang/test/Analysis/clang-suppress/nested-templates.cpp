// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify %s

void clang_analyzer_warnIfReached();

// Placeholder types for triggering instantiations.
// - Type{A,B,C,D} should match an unconstrained template type parameter.
struct TypeA{};
struct TypeB{};
struct TypeC{};
struct TypeD{};

// ============================================================================
// Group A: 2-level nesting — attribute on outer
// ============================================================================

template <typename A>
struct [[clang::suppress]] TwoLevel_AttrOuter {
  template <typename B>
  struct Inner {
    void inline_method() {
      clang_analyzer_warnIfReached(); // no-warning
    }
    void outline_method();
  };
  void outer_inline() {
    clang_analyzer_warnIfReached(); // no-warning
  }
  void outer_outline();
};

template <typename A>
template <typename B>
void TwoLevel_AttrOuter<A>::Inner<B>::outline_method() {
  // Out-of-line: lexical context is namespace, not the class.
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

template <typename A>
void TwoLevel_AttrOuter<A>::outer_outline() {
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

void test_two_level_outer() {
  TwoLevel_AttrOuter<TypeA>::Inner<TypeB>().inline_method();
  TwoLevel_AttrOuter<TypeA>::Inner<TypeB>().outline_method();
  TwoLevel_AttrOuter<TypeA>().outer_inline();
  TwoLevel_AttrOuter<TypeA>().outer_outline();
}

// ============================================================================
// Group B: 2-level nesting — attribute on inner
// ============================================================================

template <typename A>
struct TwoLevel_AttrInner {
  template <typename B>
  struct [[clang::suppress]] Inner {
    void inline_method() {
      clang_analyzer_warnIfReached(); // no-warning
    }
    void outline_method();
  };
  void outer_inline() {
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  }
};

template <typename A>
template <typename B>
void TwoLevel_AttrInner<A>::Inner<B>::outline_method() {
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}

void test_two_level_inner() {
  TwoLevel_AttrInner<TypeA>::Inner<TypeB>().inline_method();
  TwoLevel_AttrInner<TypeA>::Inner<TypeB>().outline_method();
  TwoLevel_AttrInner<TypeA>().outer_inline();
}

// ============================================================================
// Group C: 3-level nesting — attribute at each level
// ============================================================================

// --- C1: attribute on outermost ---

template <typename A>
struct [[clang::suppress]] ThreeLevel_AttrOuter {
  template <typename B>
  struct Mid {
    template <typename C>
    struct Inner {
      void method() {
        clang_analyzer_warnIfReached(); // no-warning
      }
    };
    void mid_method() {
      clang_analyzer_warnIfReached(); // no-warning
    }
  };
};

void test_three_level_outer() {
  ThreeLevel_AttrOuter<TypeA>::Mid<TypeB>::Inner<TypeC>().method();
  ThreeLevel_AttrOuter<TypeA>::Mid<TypeB>().mid_method();
}

// --- C2: attribute on middle ---

template <typename A>
struct ThreeLevel_AttrMid {
  template <typename B>
  struct [[clang::suppress]] Mid {
    template <typename C>
    struct Inner {
      void method() {
        clang_analyzer_warnIfReached(); // no-warning
      }
    };
    void mid_method() {
      clang_analyzer_warnIfReached(); // no-warning
    }
  };
  void outer_method() {
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  }
};

void test_three_level_mid() {
  ThreeLevel_AttrMid<TypeA>::Mid<TypeB>::Inner<TypeC>().method();
  ThreeLevel_AttrMid<TypeA>::Mid<TypeB>().mid_method();
  ThreeLevel_AttrMid<TypeA>().outer_method();
}

// --- C3: attribute on innermost ---

template <typename A>
struct ThreeLevel_AttrInner {
  template <typename B>
  struct Mid {
    template <typename C>
    struct [[clang::suppress]] Inner {
      void method() {
        clang_analyzer_warnIfReached(); // no-warning
      }
    };
    void mid_method() {
      clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
    }
  };
  void outer_method() {
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  }
};

void test_three_level_inner() {
  ThreeLevel_AttrInner<TypeA>::Mid<TypeB>::Inner<TypeC>().method();
  ThreeLevel_AttrInner<TypeA>::Mid<TypeB>().mid_method();
  ThreeLevel_AttrInner<TypeA>().outer_method();
}

// --- C4: no attribute at any level (negative test) ---

template <typename A>
struct ThreeLevel_NoAttr {
  template <typename B>
  struct Mid {
    template <typename C>
    struct Inner {
      void method() {
        clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
      }
    };
    void mid_method() {
      clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
    }
  };
  void outer_method() {
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  }
};

void test_three_level_none() {
  ThreeLevel_NoAttr<TypeA>::Mid<TypeB>::Inner<TypeC>().method();
  ThreeLevel_NoAttr<TypeA>::Mid<TypeB>().mid_method();
  ThreeLevel_NoAttr<TypeA>().outer_method();
}

// ============================================================================
// Group D: Mixed template / non-template nesting
// ============================================================================

// --- D1: non-template outer, template inner ---

struct [[clang::suppress]] NonTmplOuter_TmplInner {
  template <typename T>
  struct Inner {
    void method() {
      clang_analyzer_warnIfReached(); // no-warning
    }
  };
};

struct NonTmplOuter_TmplInner_NoAttr {
  template <typename T>
  struct Inner {
    void method() {
      clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
    }
  };
};

void test_mixed_nontmpl_outer() {
  NonTmplOuter_TmplInner::Inner<TypeA>().method();
  NonTmplOuter_TmplInner_NoAttr::Inner<TypeA>().method();
}

// --- D2: template outer, non-template inner ---

template <typename T>
struct [[clang::suppress]] TmplOuter_NonTmplInner {
  struct Inner {
    void method() {
      clang_analyzer_warnIfReached(); // no-warning
    }
  };
};

template <typename T>
struct TmplOuter_NonTmplInner_NoAttr {
  struct Inner {
    void method() {
      clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
    }
  };
};

void test_mixed_tmpl_outer() {
  TmplOuter_NonTmplInner<TypeA>::Inner().method();
  TmplOuter_NonTmplInner_NoAttr<TypeA>::Inner().method();
}

// ============================================================================
// Group E: Multiple instantiations of the same nested template
// ============================================================================

// Ensure suppression applies across different instantiation parameters.

template <typename A>
struct [[clang::suppress]] MultiInst {
  template <typename B>
  struct Inner {
    void method() {
      clang_analyzer_warnIfReached(); // no-warning
    }
  };
};

void test_multi_inst() {
  MultiInst<TypeA>::Inner<TypeA>().method();
  MultiInst<TypeA>::Inner<TypeB>().method();
  MultiInst<TypeB>::Inner<TypeA>().method();
  MultiInst<TypeC>::Inner<TypeD>().method();
}

// ============================================================================
// Group F: Nested template with methods that have their own template params
// ============================================================================

template <typename A>
struct [[clang::suppress]] NestedWithTemplateMethods {
  template <typename B>
  struct Inner {
    template <typename C>
    void tmpl_method(C) {
      clang_analyzer_warnIfReached(); // no-warning
    }
  };
};

template <typename A>
struct NestedWithTemplateMethods_NoAttr {
  template <typename B>
  struct Inner {
    template <typename C>
    void tmpl_method(C) {
      clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
    }
  };
};

void test_nested_tmpl_methods() {
  NestedWithTemplateMethods<TypeA>::Inner<TypeB>().tmpl_method(TypeC{});
  NestedWithTemplateMethods<TypeA>::Inner<TypeB>().tmpl_method(TypeD{});
  NestedWithTemplateMethods_NoAttr<TypeA>::Inner<TypeB>().tmpl_method(TypeC{});
}

// ============================================================================
// Group G: Attribute on both outer and inner (redundant but should work)
// ============================================================================

template <typename A>
struct [[clang::suppress]] BothSuppressed {
  template <typename B>
  struct [[clang::suppress]] Inner {
    void method() {
      clang_analyzer_warnIfReached(); // no-warning
    }
  };
  void outer_method() {
    clang_analyzer_warnIfReached(); // no-warning
  }
};

void test_both_suppressed() {
  BothSuppressed<TypeA>::Inner<TypeB>().method();
  BothSuppressed<TypeA>().outer_method();
}

// ============================================================================
// Regression test for gh#182659
// ============================================================================

// Nested template structs where the inner method accesses a member of a
// different specialization — verifies that the suppression mechanism does not
// accidentally suppress legitimate warnings when walking instantiation chains.

template <class> struct gh_182659_s1 {
  template <class> struct gh_182659_s2 {
    int i;
    template <class T> int m(const gh_182659_s2<T>& s2) {
      return s2.i; // expected-warning{{Undefined or garbage value returned to caller}}
    }
  };
};

void gh_182659() {
  gh_182659_s1<TypeA>::gh_182659_s2<TypeA> s1;
  gh_182659_s1<TypeA>::gh_182659_s2<TypeB> s2;
  s1.m(s2);
}
