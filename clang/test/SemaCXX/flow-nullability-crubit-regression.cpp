// Regression tests ported from Google Crubit's nullability checker test suite.
// Each section maps to a specific Crubit test file, testing that nullable-clang
// handles the same patterns (and more). The final sections document where we
// exceed Crubit and where we have permanent gaps.
//
// RUN: %clang_cc1 -fsyntax-only -fflow-sensitive-nullability -fnullability-default=nullable -std=c++17 -Wno-unused-value -Wno-nonnull %s -verify

#pragma clang assume_nonnull begin

// Forward declarations used throughout.
int *_Nullable GetNullable();
int *_Nonnull GetNonnull();
int *GetUnknown();
bool cond();
[[noreturn]] void fatal(const char *msg);
[[noreturn]] void abort_fn();

// ==========================================================================
// BASIC: dereference, assignment, return, argument passing
// (from crubit/nullability/test/basic.cc)
// ==========================================================================

// --- Deref nullptr ---
void test_deref_nullptr() {
  int *_Nullable x = nullptr;
  (void)*x; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

// --- Deref address-of is always safe ---
void test_deref_addr_of() {
  int i;
  int *x = &i;
  (void)*x; // no warning
}

// --- Deref address-of, transitive ---
void test_deref_addr_of_transitive() {
  int i;
  int *x = &i;
  int *y = x;
  (void)*y; // no warning
}

// --- Deref nonnull param ---
void test_deref_nonnull_param(int *_Nonnull x) {
  (void)*x; // no warning
}

// --- Deref nonnull param, transitive ---
void test_deref_nonnull_transitive(int *_Nonnull x) {
  int *y = x;
  (void)*y; // no warning
}

// --- Deref nullable param without check ---
void test_deref_nullable_unguarded(int *_Nullable x) {
  (void)*x; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

// --- Deref nullable param, transitive ---
void test_deref_nullable_transitive(int *_Nullable x) {
  int *y = x;
  (void)*y; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

// --- Arrow operator on nullable ---
struct Foo {
  int val;
  Foo *next;
  Foo *getNext();
};

void test_arrow_nullable_field(Foo *_Nullable f) {
  (void)f->val; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
  if (f) {
    (void)f->val; // no warning -- narrowed
  }
}

void test_arrow_nullable_method(Foo *_Nullable f) {
  (void)f->getNext(); // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
  if (f) {
    (void)f->getNext(); // no warning
  }
}

// --- Arrow on nonnull is always safe ---
void test_arrow_nonnull(Foo *_Nonnull f) {
  (void)f->val;       // no warning
  (void)f->getNext(); // no warning
}

// --- Array subscript (p[n] is *(p+n)) ---
void test_subscript_nullable(int *_Nonnull nonnull, int *_Nullable nullable) {
  (void)nonnull[0];  // no warning
  (void)nullable[0]; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

// --- Assignment: nullable to nonnull ---
void test_assign_nullable_to_nonnull(int *_Nullable nullable) {
  int *_Nonnull nn = GetNonnull();
  nn = nullable; // expected-warning{{assigning nullable pointer to nonnull variable}} expected-note{{add a null check}}
}

// --- Assignment: safe cases ---
void test_assign_safe(int *_Nonnull nonnull) {
  int *_Nullable x = nullptr;  // no warning
  int *_Nonnull y = nonnull;   // no warning
}

// --- Argument passing: nullable to nonnull parameter ---
void takes_nonnull(int *_Nonnull p);

void test_pass_nullable_to_nonnull(int *_Nullable p) {
  takes_nonnull(p); // expected-warning{{passing nullable pointer to nonnull parameter}} expected-note{{add a null check}}
}

void test_pass_nonnull_ok(int *_Nonnull p) {
  takes_nonnull(p); // no warning
}

// --- Return: nullable from nonnull return type ---
int *_Nonnull test_return_nullable_param(int *_Nullable p) {
  return p; // expected-warning{{returning nullable pointer from function with nonnull return type}} expected-note{{add a null check}}
}

int *_Nonnull test_return_nonnull_ok(int *_Nonnull p) {
  return p; // no warning
}

// --- Return nullable from nullable return type (always ok) ---
int *_Nullable test_return_null_from_nullable() {
  return nullptr; // no warning
}

// --- Multiple returns: one path safe, one not ---
int *_Nonnull test_return_multiple(bool b, int *_Nonnull nn) {
  if (b) {
    return GetNullable(); // expected-warning{{returning nullable pointer from function with nonnull return type}} expected-note{{add a null check}}
  }
  return nn; // no warning
}

// --- Return after null-check (narrowed) ---
int *_Nonnull test_return_narrowed(int *_Nullable p, int *_Nonnull fallback) {
  if (p) {
    return p; // no warning -- p is narrowed to nonnull
  }
  return fallback;
}

// ==========================================================================
// POINTER ARITHMETIC
// (from crubit/nullability/test/pointer_arithmetic_diagnosis.cc)
// ==========================================================================

// --- Arithmetic on nullable warns ---
void test_ptr_arith_nullable(int *_Nullable nullable, int i) {
  int *orig = nullable;

  nullable + i;  // expected-warning{{pointer arithmetic on nullable pointer}} expected-note{{add a null check before performing arithmetic}}
  nullable - i;  // expected-warning{{pointer arithmetic on nullable pointer}} expected-note{{add a null check before performing arithmetic}}

  nullable++;    // expected-warning{{pointer arithmetic on nullable pointer}} expected-note{{add a null check before performing arithmetic}}
  nullable = orig;

  ++nullable;    // expected-warning{{pointer arithmetic on nullable pointer}} expected-note{{add a null check before performing arithmetic}}
  nullable = orig;

  nullable--;    // expected-warning{{pointer arithmetic on nullable pointer}} expected-note{{add a null check before performing arithmetic}}
  nullable = orig;

  --nullable;    // expected-warning{{pointer arithmetic on nullable pointer}} expected-note{{add a null check before performing arithmetic}}
  nullable = orig;

  nullable += 1; // expected-warning{{pointer arithmetic on nullable pointer}} expected-note{{add a null check before performing arithmetic}}
  nullable = orig;

  nullable -= 1; // expected-warning{{pointer arithmetic on nullable pointer}} expected-note{{add a null check before performing arithmetic}}
}

// --- Arithmetic on nonnull is safe ---
void test_ptr_arith_nonnull(int *_Nonnull nonnull, int i) {
  int *orig = nonnull;
  nonnull + i;
  nonnull - i;
  nonnull++;
  nonnull = orig;
  ++nonnull;
  nonnull = orig;
  nonnull--;
  nonnull = orig;
  --nonnull;
  nonnull = orig;
  nonnull += 1;
  nonnull = orig;
  nonnull -= 1;
  // no warnings anywhere
}

// --- Arithmetic on nullable after null-check is safe ---
void test_ptr_arith_after_check(int *_Nullable nullable) {
  if (nullable) {
    nullable + 1; // no warning -- narrowed
    nullable++;   // no warning
  }
}

// ==========================================================================
// PATH-SENSITIVE: null checks suppress warnings, narrowing
// (from crubit/nullability/test/path_sensitive.cc)
// ==========================================================================

// --- Basic if-check narrows ---
void test_if_narrows(int *_Nullable p) {
  if (p) {
    (void)*p; // no warning
  }
  (void)*p; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

// --- if/else narrowing ---
void test_if_else(int *_Nullable p) {
  if (p) {
    (void)*p; // no warning -- true branch
  } else {
    (void)*p; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
  }
}

// --- Early return narrows ---
void test_early_return(int *_Nullable p) {
  if (!p)
    return;
  (void)*p; // no warning
}

// --- Ternary narrows ---
int test_ternary(int *_Nullable p) {
  return p ? *p : 0; // no warning -- p is checked
}

// --- != nullptr narrows ---
void test_ne_nullptr(int *_Nullable p) {
  if (p != nullptr) {
    (void)*p; // no warning
  }
}

// --- == nullptr + early return ---
void test_eq_nullptr_return(int *_Nullable p) {
  if (p == nullptr)
    return;
  (void)*p; // no warning
}

// --- ComplexLoopCondition: compound && with assignment in while ---
void test_complex_loop_condition() {
  int *p1;
  int *p2;
  while ((p1 = GetNullable()) != nullptr && (p2 = GetNullable()) != nullptr) {
    (void)*p1; // no warning -- checked in condition
    (void)*p2; // no warning -- checked in condition
  }
}

// --- For-loop can't prove body executes ---
void test_for_loop_no_guarantee() {
  int *_Nullable p = nullptr;
  int x = 0;
  for (int i = 0; i < 10; ++i) {
    p = &x;
  }
  *p = 1; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

// --- Do-while guarantees at least one iteration ---
void test_do_while_guarantees_execution() {
  int *_Nullable p = nullptr;
  int x = 0;
  int i = 0;
  do {
    p = &x;
    ++i;
  } while (i < 10);
  *p = 1; // no warning -- do-while body always executes
}

// --- ConditionalInitialization2: bool guard does not imply nonnull ---
void test_conditional_init_unsafe() {
  int *_Nullable p = nullptr;
  bool b = false;
  b = cond();
  if (!b)
    p = GetNonnull();
  (void)*p; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
}

// ==========================================================================
// ALIASES: y = x; if (y) *x works
// (from crubit/nullability/test/variable_aliasing.cc)
// ==========================================================================

// --- Check alias, deref original ---
void test_alias_check_deref_original(int *_Nullable x) {
  int *y = x;
  if (y) {
    (void)*x; // no warning -- y aliases x, y is checked
  } else {
    (void)*x; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
  }
}

// --- Check original, deref alias ---
void test_alias_check_original_deref_alias(int *_Nullable x) {
  int *y = x;
  if (x) {
    (void)*y; // no warning -- x is checked, y aliases x
  } else {
    (void)*y; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
  }
}

// ==========================================================================
// RETURN STATEMENTS (additional patterns)
// (from crubit/nullability/test/return_statements.cc)
// ==========================================================================

// --- Return from merged paths: one path null, one nonnull ---
int *_Nonnull test_return_merged_paths(bool b, int i) {
  int *_Nullable ptr;
  if (b) {
    ptr = &i;
  } else {
    ptr = nullptr;
  }
  return ptr; // expected-warning{{returning nullable pointer from function with nonnull return type}} expected-note{{add a null check}}
}

// --- Return nullable after narrowing (safe) ---
int *_Nonnull test_return_nullable_narrowed(int *_Nullable p1,
                                            int *_Nonnull fallback) {
  if (p1) {
    return p1; // no warning -- narrowed
  }
  return fallback;
}

// ==========================================================================
// CHECK MACROS: if (!p) abort(), assert(p) style
// (from crubit/nullability/test/check_macros.cc and basic.cc)
// ==========================================================================

// --- Simple CHECK macro (if + __builtin_abort) ---
#define CHECK(x) \
  if (!(x))      \
    __builtin_abort();

void test_check_macro(int *_Nullable p) {
  CHECK(p);
  (void)*p; // no warning -- CHECK asserted nonnull
}

// --- CHECK with noreturn function ---
#define ASSERT(cond) \
  do {               \
    if (!(cond))     \
      fatal("fail"); \
  } while (0)

void test_assert_macro(int *_Nullable p) {
  ASSERT(p);
  (void)*p; // no warning
}

// --- CHECK two variables ---
void test_check_two_vars(int *_Nullable p, int *_Nullable q) {
  CHECK(p);
  CHECK(q);
  (void)(*p + *q); // no warning
}

// --- if (!p) abort(); explicit pattern ---
void test_if_abort(int *_Nullable p) {
  if (!p)
    abort_fn();
  (void)*p; // no warning
}

// ==========================================================================
// CONVERGENCE: loops with nullable pointers
// (from crubit/nullability/test/convergence.cc)
// ==========================================================================

// --- Loop: nullable init, nonnull update still warns ---
void test_loop_nullable_nonnull() {
  for (int *p = GetNullable();; p = GetNonnull()) {
    (void)*p; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
  }
}

// --- Loop: both nullable ---
void test_loop_nullable_nullable() {
  for (int *p = GetNullable();; p = GetNullable()) {
    (void)*p; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
  }
}

// --- Loop: nonnull init, nullable update ---
void test_loop_nonnull_nullable() {
  for (int *p = GetNonnull();; p = GetNullable()) {
    (void)*p; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
  }
}

// --- Loop: both nonnull (safe) ---
void test_loop_nonnull_nonnull() {
  for (int *p = GetNonnull();; p = GetNonnull()) {
    (void)*p; // no warning
  }
}

// --- Loop with null check in condition (safe) ---
void test_loop_checked() {
  for (int *p = GetNullable(); p != nullptr; p = GetNullable()) {
    (void)*p; // no warning -- loop condition checks p
  }
}

// --- Loop with unrelated condition ---
void test_loop_unrelated_condition() {
  for (int *p = GetNonnull(); cond(); p = GetNullable()) {
    (void)*p; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
  }
}

// --- While assignment: while ((p = f())) ---
void test_while_assignment() {
  int *p;
  while ((p = GetNullable())) {
    (void)*p; // no warning -- loop condition checks p
  }
}

// --- InconsistentLoopStateRepro (Crubit: b/300979650) ---
// A prior loop must not corrupt subsequent null-checked code.
void test_inconsistent_loop_state(int *b, int *e) {
  for (; b != e; ++b)
    ;
  int *ptr = GetNullable();
  if (ptr != nullptr) {
    while (cond()) {
      (void)*ptr; // no warning -- ptr is checked
    }
  }
}

// --- ReproForFalsePositiveTriggeredByUnrelatedLoop ---
struct Node {
  const Node *_Nonnull parent() const;
};

void test_unrelated_loop_no_false_positive(const Node *_Nonnull node) {
  for (bool b2 = cond(); cond(); b2 = false) {
  }
  while (cond()) {
    node = node->parent(); // no warning -- node is _Nonnull
  }
}

// --- WidenAfterContradiction ---
void test_widen_after_contradiction_var() {
  bool b = true;
  while (b) {
    b = cond();
  }
  int *p = GetUnknown();
  while (cond()) {
    (void)*p; // no warning -- p is unknown (not nullable)
  }
}

void test_widen_after_contradiction_arbitrary() {
  bool b = true;
  while (cond()) {
    b = false;
  }
  if (b)
    return;
  int *p = GetUnknown();
  while (cond()) {
    (void)*p; // no warning
  }
}

// --- TriplyNestedForLoopSingleIteration ---
// Minimized from ABSL_LOG_INTERNAL_STATEFUL_CONDITION.
void test_triply_nested_loop() {
  for (bool b = true; b;)
    for (int x = 0; b;)
      for (int c = 0; b; b = false) {
        (void)0;
      }
}

// ==========================================================================
// PATTERNS NULLABLE-CLANG HANDLES THAT CRUBIT DOESN'T
// These demonstrate advantages of the CFG-based approach over Crubit's
// dataflow framework.
// ==========================================================================

// --- Aliases: bidirectional narrowing propagation ---
// Crubit handles this too, but our implementation tracks alias chains
// (y -> x -> canonical) and invalidates on reassignment.
void test_alias_chain(int *_Nullable x) {
  int *y = x;
  int *z = y; // z -> y -> x
  if (z) {
    (void)*x; // no warning -- z aliases x transitively
    (void)*y; // no warning
  }
}

// --- __builtin_expect / LIKELY / UNLIKELY macros ---
// Crubit requires special modeling for each macro. Our analysis sees
// through __builtin_expect transparently because the CFG decomposes it.
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)

void test_likely_narrowing(int *_Nullable p) {
  if (LIKELY(p)) {
    (void)*p; // no warning -- sees through __builtin_expect
  }
}

void test_unlikely_early_return(int *_Nullable p) {
  if (UNLIKELY(!p))
    return;
  (void)*p; // no warning
}

// --- __builtin_assume ---
// Crubit has no equivalent; our analysis treats __builtin_assume(p) as
// an unconditional narrowing hint.
void test_builtin_assume(int *_Nullable p) {
  __builtin_assume(p != nullptr);
  (void)*p; // no warning
}

// --- Bare-brace assertion macros ---
// Patterns like `{ if (!(p)) abort(); }` that don't use do-while.
// Our CFG-based approach handles any terminating pattern naturally.
#define BRACE_ASSERT(cond) \
  {                        \
    if (!(cond))           \
      fatal("assert");     \
  }

void test_brace_assert(int *_Nullable p) {
  BRACE_ASSERT(p);
  (void)*p; // no warning
}

// --- Pointer arithmetic warning (Crubit has this too) ---
// We emit a distinct warning group (-Wflow-nullable-arithmetic) so users
// can enable/disable pointer arithmetic checks independently.
void test_arith_warning_group(int *_Nullable p) {
  p + 1; // expected-warning{{pointer arithmetic on nullable pointer}} expected-note{{add a null check before performing arithmetic}}
}

// ==========================================================================
// PERMANENT GAPS: patterns requiring SAT-solver disjunctive reasoning
// Crubit can prove these safe; nullable-clang cannot (by design).
// ==========================================================================

// --- Disjunctive reasoning: if (!p1 && !p2) return ---
// After the early return, at least one of p1/p2 is non-null, but we can't
// determine WHICH one without a SAT solver. In the else branch below,
// p1 is null so p2 must be non-null -- but we don't track that.
void test_disjunctive_gap(int *_Nullable p1, int *_Nullable p2) {
  if (!p1 && !p2)
    return;
  if (p1)
    (void)*p1; // no warning -- p1 is checked
  else
    (void)*p2; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
               // Crubit would NOT warn here (SAT-based disjunctive reasoning).
               // This is a fundamental limitation of our set-intersection approach.
}

#pragma clang assume_nonnull end
