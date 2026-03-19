// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify %s

void clang_analyzer_warnIfReached();

// Systematic tests for [[clang::suppress]] on classes with friend declarations.
//
// Pruned matrix of valid combinations:
//   Axis 1: fwd-decl at namespace scope (yes / no)
//   Axis 2: body location (inline / out-of-line)
//   Axis 3: template (yes / no)
//
// Each case has a suppressed variant (class has [[clang::suppress]])
// and an unsuppressed variant (class without it).

// Placeholder types for triggering instantiations.
// - Type{A,B} should match an unconstrained template type parameter.
// - Specialized should match some specialization pattern.
struct TypeA{};
struct TypeB{};
struct Specialized{};

// ============================================================================
// Group A: Non-template friend functions
// ============================================================================

// --- A1: no fwd-decl, inline body ---

struct [[clang::suppress]] A1_Suppressed {
  friend void a1_suppressed(A1_Suppressed) {
    clang_analyzer_warnIfReached(); // no-warning
  }
};
struct A1_Unsuppressed {
  friend void a1_unsuppressed(A1_Unsuppressed) {
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  }
};
void test_A1() {
  a1_suppressed(A1_Suppressed{});
  a1_unsuppressed(A1_Unsuppressed{});
}

// --- A2: no fwd-decl, out-of-line body ---

struct [[clang::suppress]] A2_Suppressed {
  friend void a2_suppressed(A2_Suppressed);
};
void a2_suppressed(A2_Suppressed) {
  // Out-of-line: lexical parent is the translation unit, NOT the class.
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}
struct A2_Unsuppressed {
  friend void a2_unsuppressed(A2_Unsuppressed);
};
void a2_unsuppressed(A2_Unsuppressed) {
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}
void test_A2() {
  a2_suppressed(A2_Suppressed{});
  a2_unsuppressed(A2_Unsuppressed{});
}

// --- A3: fwd-decl, inline body ---

extern void a3_suppressed();
struct [[clang::suppress]] A3_Suppressed {
  friend void a3_suppressed() {
    clang_analyzer_warnIfReached(); // no-warning
  }
};
extern void a3_unsuppressed();
struct A3_Unsuppressed {
  friend void a3_unsuppressed() {
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  }
};
void test_A3() {
  a3_suppressed();
  a3_unsuppressed();
}

// --- A4: fwd-decl, out-of-line body ---

extern void a4_suppressed();
struct [[clang::suppress]] A4_Suppressed {
  friend void a4_suppressed();
};
void a4_suppressed() {
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}
extern void a4_unsuppressed();
struct A4_Unsuppressed {
  friend void a4_unsuppressed();
};
void a4_unsuppressed() {
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}
void test_A4() {
  a4_suppressed();
  a4_unsuppressed();
}

// ============================================================================
// Group B: Friend function templates (primary template)
// ============================================================================

// --- B1: no fwd-decl, inline body ---

struct [[clang::suppress]] B1_Suppressed {
  template <typename T>
  friend void b1_suppressed(B1_Suppressed, T) {
    clang_analyzer_warnIfReached(); // no-warning
  }
};
struct B1_Unsuppressed {
  template <typename T>
  friend void b1_unsuppressed(B1_Unsuppressed, T) {
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  }
};
void test_B1() {
  b1_suppressed(B1_Suppressed{}, TypeA{});
  b1_unsuppressed(B1_Unsuppressed{}, TypeA{});
}

// --- B2: no fwd-decl, out-of-line body ---

struct [[clang::suppress]] B2_Suppressed {
  template <typename T>
  friend void b2_suppressed(B2_Suppressed, T);
};
template <typename T>
void b2_suppressed(B2_Suppressed, T) {
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}
struct B2_Unsuppressed {
  template <typename T>
  friend void b2_unsuppressed(B2_Unsuppressed, T);
};
template <typename T>
void b2_unsuppressed(B2_Unsuppressed, T) {
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}
void test_B2() {
  b2_suppressed(B2_Suppressed{}, TypeA{});
  b2_unsuppressed(B2_Unsuppressed{}, TypeA{});
}

// --- B3: fwd-decl, inline body ---

template <typename T>
extern void b3_suppressed(T);
struct [[clang::suppress]] B3_Suppressed {
  template <typename T>
  friend void b3_suppressed(T) {
    // FIXME: This should be suppressed.
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  }
};
template <typename T>
extern void b3_unsuppressed(T);
struct B3_Unsuppressed {
  template <typename T>
  friend void b3_unsuppressed(T) {
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  }
};
void test_B3() {
  b3_suppressed(TypeA{});
  b3_unsuppressed(TypeA{});
}

// --- B4: fwd-decl, out-of-line body ---

template <typename T>
extern void b4_suppressed(T);
struct [[clang::suppress]] B4_Suppressed {
  template <typename T>
  friend void b4_suppressed(T);
};
template <typename T>
void b4_suppressed(T) {
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}
template <typename T>
extern void b4_unsuppressed(T);
struct B4_Unsuppressed {
  template <typename T>
  friend void b4_unsuppressed(T);
};
template <typename T>
void b4_unsuppressed(T) {
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}
void test_B4() {
  b4_suppressed(TypeA{});
  b4_unsuppressed(TypeA{});
}

// ============================================================================
// Group C: Friend function template explicit specializations
// ============================================================================

// --- C1: primary inline in suppressed class, explicit spec defined out-of-line ---
// The explicit specialization is NOT defined inside the class, so it should
// NOT be suppressed.

struct [[clang::suppress]] C1_Suppressed {
  template <typename T>
  friend void c1_suppressed(C1_Suppressed, T) {
    clang_analyzer_warnIfReached(); // no-warning
  }
};
template <>
void c1_suppressed(C1_Suppressed, Specialized) {
  // Explicit specialization defined outside the class — not suppressed.
  clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
}
void test_C1() {
  c1_suppressed(C1_Suppressed{}, TypeA{});         // uses primary (suppressed)
  c1_suppressed(C1_Suppressed{}, Specialized{});   // uses explicit spec (not suppressed)
}

// ============================================================================
// Group D: Friend classes (declared, not defined inline — C++ forbids
// defining a type in a friend declaration)
// ============================================================================

// --- D1: friend class only declared, defined outside ---

struct [[clang::suppress]] D1_Suppressed {
  friend struct D1_FriendOuter;
};
struct D1_FriendOuter {
  void method() {
    // Defined outside the suppressed class — not suppressed.
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  }
};
void test_D1() {
  D1_FriendOuter{}.method();
}

// ============================================================================
// Group E: Edge cases
// ============================================================================

// --- E1: friend function in suppressed CLASS TEMPLATE (not just suppressed class) ---

template <typename U>
struct [[clang::suppress]] E1_SuppressedTmpl {
  friend void e1_friend(E1_SuppressedTmpl) {
    clang_analyzer_warnIfReached(); // no-warning
  }
};
void test_E1() {
  e1_friend(E1_SuppressedTmpl<TypeA>{});
}

// --- E2: friend function template in a nested suppressed class ---
// The friend needs a parameter of the nested class type for ADL lookup.

struct Outer_E2 {
  struct [[clang::suppress]] Inner_E2 {
    template <typename T>
    friend void e2_inner_friend(Inner_E2, T) {
      clang_analyzer_warnIfReached(); // no-warning
    }
  };
};
void test_E2() {
  e2_inner_friend(Outer_E2::Inner_E2{}, TypeA{});
}

// --- E3: multiple redeclarations at namespace scope before friend decl ---

template <typename T> void e3_multi_redecl(T);
template <typename T> void e3_multi_redecl(T);
template <typename T> void e3_multi_redecl(T);
struct [[clang::suppress]] E3_Suppressed {
  template <typename T>
  friend void e3_multi_redecl(T) {
    // FIXME: This should be suppressed.
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  }
};
void test_E3() {
  e3_multi_redecl(TypeA{});
}

// --- E4: friend in anonymous namespace ---

namespace {
struct [[clang::suppress]] E4_AnonSuppressed {
  friend void e4_anon_friend(E4_AnonSuppressed) {
    clang_analyzer_warnIfReached(); // no-warning
  }
};
} // namespace
void test_E4() {
  e4_anon_friend(E4_AnonSuppressed{});
}

// --- E5: suppression on the friend declaration itself, not on the class ---
// Friend functions need a parameter for ADL visibility.

struct E5_ClassNotSuppressed {
  [[clang::suppress]] friend void e5_suppressed(E5_ClassNotSuppressed) {
    clang_analyzer_warnIfReached(); // no-warning
  }
  friend void e5_unsuppressed(E5_ClassNotSuppressed) {
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  }
};
void test_E5() {
  e5_suppressed(E5_ClassNotSuppressed{});
  e5_unsuppressed(E5_ClassNotSuppressed{});
}

// --- E6: friend function template in suppressed class template, with fwd-decl ---
// Combines class template + function template + fwd-decl.

template <typename T> void e6_combined(T);
template <typename U>
struct [[clang::suppress]] E6_SuppressedTmpl {
  template <typename T>
  friend void e6_combined(T) {
    // FIXME: This should be suppressed.
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  }
};
void test_E6() {
  // Instantiate the class template to make the friend visible.
  E6_SuppressedTmpl<TypeA> e6; // This line is IMPORTANT!
  (void)e6;
  e6_combined(TypeA{});
}

// --- E7: friend function template instantiated with multiple different types ---
// Ensure suppression applies to ALL instantiations, not just one.

struct [[clang::suppress]] E7_Suppressed {
  template <typename T>
  friend void e7_multi_inst(E7_Suppressed, T) {
    clang_analyzer_warnIfReached(); // no-warning
  }
};
void test_E7() {
  e7_multi_inst(E7_Suppressed{}, TypeA{});
  e7_multi_inst(E7_Suppressed{}, TypeB{});
}

// --- E8: friend function template with fwd-decl, instantiated with multiple types ---

template <typename T> void e8_fwd_multi(T);
struct [[clang::suppress]] E8_Suppressed {
  template <typename T>
  friend void e8_fwd_multi(T) {
    // FIXME: This should be suppressed.
    clang_analyzer_warnIfReached(); // expected-warning{{REACHABLE}}
  }
};
void test_E8() {
  e8_fwd_multi(TypeA{});
  e8_fwd_multi(TypeB{});
}
