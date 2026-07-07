// RUN: %clang_cc1 -fsyntax-only -verify=expected -fprofiles -fprofiles-test-profiles -std=c++23 %s
// RUN: %clang_cc1 -fsyntax-only -verify=no-profiles -std=c++23 %s
// RUN: %clang_cc1 -fsyntax-only -verify=no-test -fprofiles -std=c++23 %s

// Under -fprofiles without -fprofiles-test-profiles the built-in test:: profiles
// are inert: the attributes are recognized (no "ignored" warnings) but no rule
// fires anywhere in this file.
// no-test-no-diagnostics

// no-profiles-warning@+1 {{'profiles::enforce' attribute ignored}}
[[profiles::enforce(test::type_cast)]];
// no-profiles-warning@+1 {{'profiles::enforce' attribute ignored}}
[[profiles::enforce(test::other)]];

void test_violation() {
  int *p = reinterpret_cast<int*>(0); // expected-error {{'reinterpret_cast' is unsafe under profile 'test::type_cast'}}
}

// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
[[profiles::suppress(test::type_cast)]]
void test_suppress_decl() {
  int *p = reinterpret_cast<int*>(0);
}

void test_suppress_stmt() {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(test::type_cast)]] {
    int *p = reinterpret_cast<int*>(0);
  }
  int *q = reinterpret_cast<int*>(0); // expected-error {{'reinterpret_cast' is unsafe under profile 'test::type_cast'}}
}

void test_suppress_stmt_with_rule() {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(test::type_cast, rule: "reinterpret_cast")]] {
    int *p = reinterpret_cast<int*>(0);
  }
}

void test_suppress_stmt_with_bare_rule() {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(test::type_cast, rule: reinterpret_cast)]] {
    int *p = reinterpret_cast<int*>(0);
  }
}

void test_suppress_stmt_with_nonmatching_rule() {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(test::type_cast, rule: "static_cast")]] {
    int *p = reinterpret_cast<int*>(0); // expected-error {{'reinterpret_cast' is unsafe under profile 'test::type_cast'}}
  }
}

// P3589R2 [decl.attr.enforce]p2: static semantics applied after translation
// phase 7 -- no diagnostic in template definition, only at instantiation.
template <typename T>
void template_cast(T x) {
  auto *p = reinterpret_cast<int*>(x); // expected-error {{'reinterpret_cast' is unsafe under profile 'test::type_cast'}}
}
void instantiate() {
  template_cast(0); // expected-note {{in instantiation of function template specialization 'template_cast<int>' requested here}}
}

// P3589R2 Section 1.1: profile violations must not affect overload resolution.
// If the profile error in the decltype SFINAE'd out the first overload, the
// fallback (returning 1) would be selected and the static_assert would fire.
template <typename T>
auto sfinae_cast(T x) -> decltype(reinterpret_cast<int*>(x)) {
  return nullptr;
}
template <typename T>
auto sfinae_cast(...) -> int { return 1; }

static_assert(__is_same(decltype(sfinae_cast<long>(0L)), int *),
              "profile violation must not SFINAE out the first overload");

// Profile violations are suppressed in unevaluated contexts.
void test_unevaluated() {
  using T = decltype(reinterpret_cast<int*>(0));
}

// Suppress on TU-scope variable initializer (pull model via push in ParseDeclGroup).
// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
[[profiles::suppress(test::type_cast)]]
int *tu_scope_var = reinterpret_cast<int*>(0);

// Suppress on block-scope variable initializer.
void test_suppress_var_init() {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(test::type_cast)]] int *p = reinterpret_cast<int*>(0);
}

// Profile violations are suppressed in discarded if-constexpr branches.
void test_discarded_branch() {
  if constexpr (false) {
    int *p = reinterpret_cast<int*>(0);
  }
}

// Lambda inside enforced scope.
void test_lambda() {
  auto f = []() {
    int *p = reinterpret_cast<int*>(0); // expected-error {{'reinterpret_cast' is unsafe under profile 'test::type_cast'}}
  };
}

// Nested suppression with correct save/restore.
void test_nested_suppress() {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(test::type_cast)]] {
    int *p = reinterpret_cast<int*>(0);
    // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
    [[profiles::suppress(test::type_cast)]] {
      int *q = reinterpret_cast<int*>(0);
    }
    int *r = reinterpret_cast<int*>(0);
  }
  int *s = reinterpret_cast<int*>(0); // expected-error {{'reinterpret_cast' is unsafe under profile 'test::type_cast'}}
}

// sizeof is an unevaluated context.
void test_sizeof_unevaluated() {
  auto s = sizeof(reinterpret_cast<int*>(0));
}

// noexcept is an unevaluated context.
void test_noexcept_unevaluated() {
  bool b = noexcept(reinterpret_cast<int*>(0));
}

// Requires-expression is an unevaluated context.
void test_requires_unevaluated() {
  bool b = requires { reinterpret_cast<int*>(0); };
}

// Default function argument with violation.
void default_arg_func(int *p = reinterpret_cast<int*>(0)); // expected-error {{'reinterpret_cast' is unsafe under profile 'test::type_cast'}}

// Suppress with justification works identically to suppress without.
// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
[[profiles::suppress(test::type_cast, justification: "legacy code")]]
void test_suppress_with_justification() {
  int *p = reinterpret_cast<int*>(0);
}

// Suppress on various statement kinds.
void test_suppress_on_stmts() {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(test::type_cast)]]
  for (int *p = reinterpret_cast<int*>(0);;) break;

  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(test::type_cast)]]
  while (reinterpret_cast<int*>(0)) break;

  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(test::type_cast)]]
  if (reinterpret_cast<int*>(0)) {}

  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(test::type_cast)]]
  (void)reinterpret_cast<int*>(0);
}

// Suppress on null-statement is a no-op: the next statement is NOT suppressed.
void test_suppress_null_stmt() {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(test::type_cast)]];
  int *p = reinterpret_cast<int*>(0); // expected-error {{'reinterpret_cast' is unsafe under profile 'test::type_cast'}}
}

// Suppress on class definition: member functions are suppressed via DeclAttr.
// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
struct [[profiles::suppress(test::type_cast)]] SuppressedStruct {
  void f() {
    int *p = reinterpret_cast<int*>(0);
  }
};

// Suppress on namespace: functions inside are suppressed via DeclAttr.
// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
namespace [[profiles::suppress(test::type_cast)]] suppressed_ns {
  void g() {
    int *p = reinterpret_cast<int*>(0);
  }
}

// Selective suppression: suppressing a different profile does not
// suppress test::type_cast violations.
void test_selective_suppress() {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(test::other)]] {
    int *p = reinterpret_cast<int*>(0); // expected-error {{'reinterpret_cast' is unsafe under profile 'test::type_cast'}}
  }
}

// Suppress on compound statement inside template: suppression must be
// effective during instantiation, not just during parsing.
template <typename T>
void template_suppress_stmt(T x) {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(test::type_cast)]] {
    auto *p = reinterpret_cast<int*>(x);
  }
}
void instantiate_suppress_stmt() { template_suppress_stmt(0); }

// Suppress on variable declaration inside template.
template <typename T>
void template_suppress_var(T x) {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(test::type_cast)]] auto *p = reinterpret_cast<int*>(x);
}
void instantiate_suppress_var() { template_suppress_var(0); }

// Suppress ends at statement boundary inside template.
template <typename T>
void template_suppress_boundary(T x) {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(test::type_cast)]] {
    auto *p = reinterpret_cast<int*>(x);
  }
  auto *q = reinterpret_cast<int*>(x); // expected-error {{'reinterpret_cast' is unsafe under profile 'test::type_cast'}}
}
void instantiate_suppress_boundary() {
  template_suppress_boundary(0); // expected-note {{in instantiation of function template specialization 'template_suppress_boundary<int>' requested here}}
}

// Suppress on forward declaration does not propagate to definition.
// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
[[profiles::suppress(test::type_cast)]]
void suppress_fwd_only();

void suppress_fwd_only() {
  int *p = reinterpret_cast<int*>(0); // expected-error {{'reinterpret_cast' is unsafe under profile 'test::type_cast'}}
}

// Suppress on field NSDMI in non-template class: suppression must be
// effective during late-parsing of the in-class initializer.
struct FieldSuppressNonTemplate {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(test::type_cast)]] int *p = reinterpret_cast<int*>(0);
};

// Suppress on field NSDMI in class template: suppression must carry through
// when the in-class initializer is instantiated.
template <typename T>
struct FieldSuppressTemplate {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(test::type_cast)]] T *p = reinterpret_cast<T*>(0);
};
FieldSuppressTemplate<int> field_suppress_inst;

// Without suppress on field, the NSDMI violation fires during instantiation.
template <typename T>
struct FieldNoSuppressTemplate { // #FieldNoSuppressTemplate
  T *p = reinterpret_cast<T*>(0); // expected-error {{'reinterpret_cast' is unsafe under profile 'test::type_cast'}} \
                                  // expected-note@#FieldNoSuppressTemplate {{in instantiation of default member initializer 'FieldNoSuppressTemplate<int>::p' requested here}}
};
FieldNoSuppressTemplate<int> field_no_suppress_inst; // expected-note {{in evaluation of exception specification for 'FieldNoSuppressTemplate<int>::FieldNoSuppressTemplate' needed here}}

// Profile violations fire in constexpr functions. Use a guarded path so the
// function can still produce a constant expression (avoiding the unrelated
// "constexpr function never produces a constant expression" error).
constexpr int *constexpr_cast(bool b) {
  if (b)
    return reinterpret_cast<int*>(0); // expected-error {{'reinterpret_cast' is unsafe under profile 'test::type_cast'}}
  return nullptr;
}

// Profile violations fire in consteval functions.
consteval int *consteval_cast(bool b) {
  if (b)
    return reinterpret_cast<int*>(0); // expected-error {{'reinterpret_cast' is unsafe under profile 'test::type_cast'}}
  return nullptr;
}

// Suppress works inside constexpr functions.
constexpr int *constexpr_suppress_cast(bool b) {
  if (b) {
    // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
    [[profiles::suppress(test::type_cast)]] return reinterpret_cast<int*>(0);
  }
  return nullptr;
}

// Trailing declarator-position suppress on TU-scope variable initializer.
// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
int *trailing_tu_var [[profiles::suppress(test::type_cast)]] = reinterpret_cast<int*>(0);

// Trailing declarator-position suppress on block-scope variable initializer.
void test_trailing_suppress_block() {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  int *p [[profiles::suppress(test::type_cast)]] = reinterpret_cast<int*>(0);
  int *q = reinterpret_cast<int*>(0); // expected-error {{'reinterpret_cast' is unsafe under profile 'test::type_cast'}}
}

// Suppress on class template definition: member functions should be suppressed
// during instantiation via DeclContext walk.
template <typename T>
// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
struct [[profiles::suppress(test::type_cast)]] SuppressedClassTemplate {
  void f() {
    T *p = reinterpret_cast<T*>(0);
  }
};
SuppressedClassTemplate<int> suppressed_class_tmpl_inst;

// Without suppress on class template: member violation fires during instantiation.
template <typename T>
struct UnsuppressedClassTemplate {
  void f() {
    T *p = reinterpret_cast<T*>(0); // expected-error {{'reinterpret_cast' is unsafe under profile 'test::type_cast'}}
  }
};
void instantiate_unsuppressed_class_tmpl() {
  UnsuppressedClassTemplate<int> x;
  x.f(); // expected-note {{in instantiation of member function 'UnsuppressedClassTemplate<int>::f' requested here}}
}

// Suppress on variable template: suppression must carry through instantiation.
template <typename T>
// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
[[profiles::suppress(test::type_cast)]] T *var_tmpl_suppress = reinterpret_cast<T*>(0);
template int *var_tmpl_suppress<int>;

// Suppress on static data member template: suppression must carry through
// instantiation even when the suppress is only on the variable, not the class.
template <typename T>
struct StaticMemberSuppressTemplate {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(test::type_cast)]] static T *p;
};
template <typename T>
// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
[[profiles::suppress(test::type_cast)]] T *StaticMemberSuppressTemplate<T>::p = reinterpret_cast<T*>(0);
template struct StaticMemberSuppressTemplate<int>;

// Out-of-line member function of a suppressed class: suppression does NOT
// extend past the class body.
// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
struct [[profiles::suppress(test::type_cast)]] OutOfLineSuppressedClass {
  void inline_ok() {
    int *p = reinterpret_cast<int*>(0);
  }
  void out_of_line();
  static int *s;
};

void OutOfLineSuppressedClass::out_of_line() {
  int *p = reinterpret_cast<int*>(0); // expected-error {{'reinterpret_cast' is unsafe under profile 'test::type_cast'}}
}

// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
[[profiles::suppress(test::type_cast)]] int *OutOfLineSuppressedClass::s = reinterpret_cast<int*>(0);

// Out-of-line function in a formerly-suppressed namespace: suppression does
// NOT extend past the namespace body.
// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
namespace [[profiles::suppress(test::type_cast)]] OutOfLineSuppressedNS {
  void inline_ok() {
    int *p = reinterpret_cast<int*>(0);
  }
  void out_of_line();
}

void OutOfLineSuppressedNS::out_of_line() {
  int *p = reinterpret_cast<int*>(0); // expected-error {{'reinterpret_cast' is unsafe under profile 'test::type_cast'}}
}

// Out-of-line member of a suppressed class template.
template <typename T>
// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
struct [[profiles::suppress(test::type_cast)]] OutOfLineSuppressedClassTemplate {
  void inline_ok() {
    T *p = reinterpret_cast<T*>(0);
  }
  void out_of_line();
};

template <typename T>
void OutOfLineSuppressedClassTemplate<T>::out_of_line() {
  T *p = reinterpret_cast<T*>(0); // expected-error {{'reinterpret_cast' is unsafe under profile 'test::type_cast'}}
}

template struct OutOfLineSuppressedClassTemplate<int>; // expected-note {{in instantiation of member function 'OutOfLineSuppressedClassTemplate<int>::out_of_line' requested here}}

// Nested suppress: class template inside a suppressed outer class.
// Suppression on Outer must propagate to inline members of Inner during
// instantiation via the lexical parent chain.
// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
struct [[profiles::suppress(test::type_cast)]] NestedSuppressOuter {
  template <typename T>
  struct Inner {
    void f() { T *p = reinterpret_cast<T*>(0); }
    void out_of_line();
  };
};

template <typename T>
void NestedSuppressOuter::Inner<T>::out_of_line() {
  T *p = reinterpret_cast<T*>(0); // expected-error {{'reinterpret_cast' is unsafe under profile 'test::type_cast'}}
}

template struct NestedSuppressOuter::Inner<int>; // expected-note {{in instantiation of member function 'NestedSuppressOuter::Inner<int>::out_of_line' requested here}}

// Nested suppress: class template inside a suppressed namespace.
// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
namespace [[profiles::suppress(test::type_cast)]] NestedSuppressNS {
  template <typename T>
  struct Inner {
    void f() { T *p = reinterpret_cast<T*>(0); }
    void out_of_line();
  };
}

template <typename T>
void NestedSuppressNS::Inner<T>::out_of_line() {
  T *p = reinterpret_cast<T*>(0); // expected-error {{'reinterpret_cast' is unsafe under profile 'test::type_cast'}}
}

template struct NestedSuppressNS::Inner<int>; // expected-note {{in instantiation of member function 'NestedSuppressNS::Inner<int>::out_of_line' requested here}}

// NSDMI in a suppressed class template: suppression applies via the lexical
// parent chain during default member initializer instantiation.
template <typename T>
// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
struct [[profiles::suppress(test::type_cast)]] SuppressedNSDMI {
  T *p = reinterpret_cast<T*>(0);
};
SuppressedNSDMI<int> suppressed_nsdmi_inst;

// Inline static data member in a suppressed class template.
template <typename T>
// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
struct [[profiles::suppress(test::type_cast)]] SuppressedInlineStatic {
  static inline T *s = reinterpret_cast<T*>(0);
};
template struct SuppressedInlineStatic<int>;

// Generic lambda defined inside a suppress block, returned, and instantiated
// outside -- the suppress must carry through instantiation.
auto get_suppressed_generic_lambda() {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(test::type_cast)]] {
    auto l = [](auto x) { return reinterpret_cast<int*>(x); };
    return l;
  }
}
void test_generic_lambda_suppress_propagation() {
  auto l = get_suppressed_generic_lambda();
  l(0);
}

// Generic lambda inside a suppress with rule restriction: only the matching
// rule is suppressed.
auto get_rule_restricted_lambda() {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(test::type_cast, rule: "reinterpret_cast")]] {
    auto l = [](auto x) { return reinterpret_cast<int*>(x); };
    return l;
  }
}
void test_generic_lambda_rule_suppress() {
  auto l = get_rule_restricted_lambda();
  l(0);
}

// Generic lambda without suppress: the violation still fires during
// instantiation.
auto get_unsuppressed_generic_lambda() {
  auto l = [](auto x) {
    return reinterpret_cast<int*>(x); // expected-error {{'reinterpret_cast' is unsafe under profile 'test::type_cast'}}
  };
  return l;
}
void test_generic_lambda_no_suppress() {
  auto l = get_unsuppressed_generic_lambda();
  l(0); // expected-note {{in instantiation of function template specialization 'get_unsuppressed_generic_lambda()::(lambda)::operator()<int>' requested here}}
}

// Non-generic lambda inside suppress: still works (regression check).
void test_nongeneric_lambda_suppress() {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(test::type_cast)]] {
    auto l = []() { return reinterpret_cast<int*>(0); };
    l();
  }
}

// Suppress of a different profile does not propagate to the generic lambda.
auto get_wrong_profile_lambda() {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(test::other)]] {
    auto l = [](auto x) {
      return reinterpret_cast<int*>(x); // expected-error {{'reinterpret_cast' is unsafe under profile 'test::type_cast'}}
    };
    return l;
  }
}
void test_wrong_profile_lambda() {
  auto l = get_wrong_profile_lambda();
  l(0); // expected-note {{in instantiation of function template specialization 'get_wrong_profile_lambda()::(lambda)::operator()<int>' requested here}}
}

// Direct suppress on a non-generic lambda's declarator applies to its body.
// The attribute precedes the parameter list so it appertains to the call
// operator declaration (P2173; C++23 standard).
void test_nongeneric_lambda_direct_suppress() {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  auto l = [] [[profiles::suppress(test::type_cast)]] () {
    int *p = reinterpret_cast<int*>(0);
  };
  l();
}

// Direct suppress of a non-matching profile on a non-generic lambda does not
// suppress the violation.
void test_nongeneric_lambda_direct_suppress_mismatch() {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  auto l = [] [[profiles::suppress(test::other)]] () {
    int *p = reinterpret_cast<int*>(0); // expected-error {{'reinterpret_cast' is unsafe under profile 'test::type_cast'}}
  };
  l();
}

// Rule-based direct suppress on a non-generic lambda applies when the rule
// matches.
void test_nongeneric_lambda_direct_suppress_rule() {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  auto l = [] [[profiles::suppress(test::type_cast, rule: "reinterpret_cast")]] () {
    int *p = reinterpret_cast<int*>(0);
  };
  l();
}

// Rule-based direct suppress on a non-generic lambda does not suppress a
// non-matching rule.
void test_nongeneric_lambda_direct_suppress_wrong_rule() {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  auto l = [] [[profiles::suppress(test::type_cast, rule: "static_cast")]] () {
    int *p = reinterpret_cast<int*>(0); // expected-error {{'reinterpret_cast' is unsafe under profile 'test::type_cast'}}
  };
  l();
}

// Suppress on an inline member function definition applies to the body.
struct InlineMethodSuppress {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(test::type_cast)]]
  void f() {
    int *p = reinterpret_cast<int*>(0);
  }
};

// Suppress on an inline member function with a non-matching profile does not
// suppress the violation.
struct InlineMethodWrongProfile {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(test::other)]]
  void f() {
    int *p = reinterpret_cast<int*>(0); // expected-error {{'reinterpret_cast' is unsafe under profile 'test::type_cast'}}
  }
};

// Suppress on an inline constructor applies to the member initializer list.
struct InlineCtorMemInitSuppress {
  int *p;
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(test::type_cast)]]
  InlineCtorMemInitSuppress() : p(reinterpret_cast<int*>(0)) {}
};

// Suppress on an inline destructor applies to the destructor body.
struct InlineDtorSuppress {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(test::type_cast)]]
  ~InlineDtorSuppress() {
    int *p = reinterpret_cast<int*>(0);
  }
};

// Suppress on an inline conversion operator applies to its body.
struct InlineConversionSuppress {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(test::type_cast)]]
  operator int *() {
    return reinterpret_cast<int*>(0);
  }
};

// Suppress on an inline operator overload applies to its body.
struct InlineOperatorSuppress {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(test::type_cast)]]
  int *operator+() {
    return reinterpret_cast<int*>(0);
  }
};

// Suppress on a late-parsed default argument of an inline member function.
struct InlineDefaultArgSuppress {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(test::type_cast)]]
  void f(int *p = reinterpret_cast<int*>(0));
};

// Without per-method suppress, the violation still fires in an inline body.
struct InlineMethodNoSuppress {
  void f() {
    int *p = reinterpret_cast<int*>(0); // expected-error {{'reinterpret_cast' is unsafe under profile 'test::type_cast'}}
  }
};

// Per-method suppress on a member function template with a non-dependent
// violation in the body: exercises getAsFunction() on a FunctionTemplateDecl.
struct InlineMemberTemplateSuppress {
  template <typename T>
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(test::type_cast)]]
  void f() {
    int *p = reinterpret_cast<int*>(0);
  }
};
void instantiate_inline_member_template_suppress() {
  InlineMemberTemplateSuppress s;
  s.f<int>();
}

// Suppress on a static inline data member applies to its in-class initializer.
struct StaticInlineDataMemberSuppress {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(test::type_cast)]]
  static inline int *p = reinterpret_cast<int*>(0);
};

// Without per-member suppress, a static inline data member initializer fires.
struct StaticInlineDataMemberNoSuppress {
  static inline int *p = reinterpret_cast<int*>(0); // expected-error {{'reinterpret_cast' is unsafe under profile 'test::type_cast'}}
};

// Suppress on a static data member initializer with a non-matching profile
// does not suppress the violation.
struct StaticInlineDataMemberWrongProfile {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(test::other)]]
  static inline int *p = reinterpret_cast<int*>(0); // expected-error {{'reinterpret_cast' is unsafe under profile 'test::type_cast'}}
};

// Suppress on a nested (non-template) class must reach its late-parsed inline
// method bodies, NSDMIs, and default arguments even though the nested class's
// body is parsed before the outer class's members are late-parsed.
struct NestedSuppressInnerOuter {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  struct [[profiles::suppress(test::type_cast)]] Inner {
    void f() {
      int *p = reinterpret_cast<int*>(0);
    }
    int *p = reinterpret_cast<int*>(0);
    void g(int *q = reinterpret_cast<int*>(0));
  };
};

// Non-matching profile on the nested class does not suppress the violation.
struct NestedSuppressWrongProfile {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  struct [[profiles::suppress(test::other)]] Inner {
    void f() {
      int *p = reinterpret_cast<int*>(0); // expected-error {{'reinterpret_cast' is unsafe under profile 'test::type_cast'}}
    }
    int *p = reinterpret_cast<int*>(0); // expected-error {{'reinterpret_cast' is unsafe under profile 'test::type_cast'}}
    void g(int *q = reinterpret_cast<int*>(0)); // expected-error {{'reinterpret_cast' is unsafe under profile 'test::type_cast'}}
  };
};

// Suppress on the outer class extends to inline bodies / NSDMIs / default
// args of a nested non-template class.
// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
struct [[profiles::suppress(test::type_cast)]] OuterSuppressNested {
  struct Inner {
    void f() {
      int *p = reinterpret_cast<int*>(0);
    }
    int *p = reinterpret_cast<int*>(0);
    void g(int *q = reinterpret_cast<int*>(0));
  };
};

// Suppress on an enclosing namespace reaches a nested non-template class.
// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
namespace [[profiles::suppress(test::type_cast)]] nested_suppress_ns {
  struct Inner {
    void f() {
      int *p = reinterpret_cast<int*>(0);
    }
    int *p = reinterpret_cast<int*>(0);
    void g(int *q = reinterpret_cast<int*>(0));
  };
}

// Deeply nested: suppress on the middle class reaches the innermost class's
// inline body.
struct DeepOuter {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  struct [[profiles::suppress(test::type_cast)]] DeepMiddle {
    struct DeepInner {
      void f() {
        int *p = reinterpret_cast<int*>(0);
      }
      int *p = reinterpret_cast<int*>(0);
      void g(int *q = reinterpret_cast<int*>(0));
    };
  };
};

// A [[profiles::suppress]] live at the point of instantiation covers the
// trigger's tokens, not the pattern's (P3589R2 s2.4p3, token-based dominion):
// parse-time checks in synchronously instantiated code must not be
// suppressed by the caller's scope.

int instantiation_leak_target = 0;

// Function-template body instantiated from a suppressed initializer.
template <typename T>
auto instantiation_leak_fn() {
  return reinterpret_cast<T*>(&instantiation_leak_target); // expected-error {{'reinterpret_cast' is unsafe under profile 'test::type_cast'}}
}
// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
[[profiles::suppress(test::type_cast)]] auto *leak_fn_use = instantiation_leak_fn<int>(); // expected-note {{in instantiation of function template specialization 'instantiation_leak_fn<int>' requested here}}

// NSDMI of an unrelated class template instantiated from a suppressed
// declaration.
template <typename T>
struct LeakNSDMI { // #LeakNSDMI
  T *p = reinterpret_cast<T*>(0); // expected-error {{'reinterpret_cast' is unsafe under profile 'test::type_cast'}} \
                                  // expected-note@#LeakNSDMI {{in instantiation of default member initializer 'LeakNSDMI<int>::p' requested here}}
};
// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
[[profiles::suppress(test::type_cast)]] LeakNSDMI<int> leak_nsdmi_use; // expected-note {{in evaluation of exception specification for 'LeakNSDMI<int>::LeakNSDMI' needed here}}

// Variable-template initializer.
template <typename T>
auto *leak_vt = reinterpret_cast<T*>(&instantiation_leak_target); // expected-error {{'reinterpret_cast' is unsafe under profile 'test::type_cast'}}
// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
[[profiles::suppress(test::type_cast)]] int *leak_vt_use = leak_vt<int>; // expected-note {{in instantiation of variable template specialization 'leak_vt' requested here}}

// Default argument instantiated at a suppressed call site.
template <typename T>
int *leak_def(T *q = reinterpret_cast<T*>(&instantiation_leak_target)) { return q; } // expected-error {{'reinterpret_cast' is unsafe under profile 'test::type_cast'}}
// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
[[profiles::suppress(test::type_cast)]] int *leak_def_use = leak_def<int>(); // expected-note {{in instantiation of default function argument expression for 'leak_def<int>' required here}}

// A closure created during the instantiation must not absorb the caller's
// suppress either.
template <typename T>
auto leak_lambda() {
  auto l = [](auto x) { return reinterpret_cast<T*>(x); }; // expected-error {{'reinterpret_cast' is unsafe under profile 'test::type_cast'}}
  return l(0L); // expected-note {{in instantiation of function template specialization 'leak_lambda()::(lambda)::operator()<long>' requested here}}
}
// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
[[profiles::suppress(test::type_cast)]] auto *leak_lambda_use = leak_lambda<int>(); // expected-note {{in instantiation of function template specialization 'leak_lambda<int>' requested here}}

// A suppressed *statement* inside a pattern does not reach a lexically
// unrelated pattern instantiated under it: GapLeaked's tokens precede the
// suppressed block, so its NSDMI check still fires.
template <typename T>
struct GapLeaked { // #GapLeaked
  T *p = reinterpret_cast<T*>(0); // expected-error {{'reinterpret_cast' is unsafe under profile 'test::type_cast'}} \
                                  // expected-note@#GapLeaked {{in instantiation of default member initializer 'GapLeaked<int>::p' requested here}}
};
template <typename T>
void gap_user() {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(test::type_cast)]] {
    GapLeaked<T> v; // expected-note {{in evaluation of exception specification for 'GapLeaked<int>::GapLeaked' needed here}}
    (void)v;
  }
}
template void gap_user<int>(); // expected-note {{in instantiation of function template specialization 'gap_user<int>' requested here}}

// Contrast: a local class *defined inside* the suppressed block is within the
// dominion, so its NSDMI stays suppressed when instantiated under it.
template <typename T>
void local_class_in_suppressed_block() {
  // no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
  [[profiles::suppress(test::type_cast)]] {
    struct Local { T *p = reinterpret_cast<T*>(0); };
    Local v;
    (void)v;
  }
}
template void local_class_in_suppressed_block<int>();

// Dominion matching compares raw TU token order, so macro-emitted code
// behaves by its expansion position: a violation spelled in a macro argument
// of the suppressed declaration is within the dominion; a pattern's
// macro-emitted violation stays outside it.
#define EMIT_CAST(ty, x) reinterpret_cast<ty*>(x)
// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
[[profiles::suppress(test::type_cast)]] int *macro_arg_suppressed = EMIT_CAST(int, &instantiation_leak_target);
template <typename T>
auto leak_macro_fn() { return EMIT_CAST(T, &instantiation_leak_target); } // expected-error {{'reinterpret_cast' is unsafe under profile 'test::type_cast'}}
// no-profiles-warning@+1 {{'profiles::suppress' attribute ignored}}
[[profiles::suppress(test::type_cast)]] int *leak_macro_use = leak_macro_fn<int>(); // expected-note {{in instantiation of function template specialization 'leak_macro_fn<int>' requested here}}
