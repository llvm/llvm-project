// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -analyzer-config c++-inlining=constructors -verify %s \
// RUN:   2>&1 | FileCheck %s


void clang_analyzer_printState();
template <typename T> void clang_analyzer_dump_lref(T& param);
template <typename T> void clang_analyzer_dump_val(T param);
template <typename T> T conjure();
template <typename... Ts> void nop(const Ts &... args) {}

struct aggr {
  int x;
  int y;
};

struct empty {
};

void test_copy_return() {
  aggr s1 = {1, 2};
  aggr const& cr1 = aggr(s1);
  clang_analyzer_dump_lref(cr1); // expected-warning-re {{&lifetime_extended_object{aggr, cr1, S{{[0-9]+}}} }}

  empty s2;
  empty const& cr2 = empty{s2};
  clang_analyzer_dump_lref(cr2); // expected-warning-re {{&lifetime_extended_object{empty, cr2, S{{[0-9]+}}} }}
}

void test_assign_return() {
  aggr s1 = {1, 2};
  aggr d1;
  clang_analyzer_dump_lref(d1 = s1); // expected-warning {{&d1 }}

  empty s2;
  empty d2;
  clang_analyzer_dump_lref(d2 = s2); // expected-warning {{&d2 }} was Unknown
}


namespace trivial_struct_copy {

void _01_empty_structs() {
  clang_analyzer_dump_val(conjure<empty>()); // expected-warning {{lazyCompoundVal}}
  empty Empty = conjure<empty>();
  empty Empty2 = Empty;
  empty Empty3 = Empty2;
  // All of these should refer to the exact same LCV, because all of
  // these trivial copies refer to the original conjured value.
  // There were Unknown before:
  clang_analyzer_dump_val(Empty);  // expected-warning {{lazyCompoundVal}}
  clang_analyzer_dump_val(Empty2); // expected-warning {{lazyCompoundVal}}
  clang_analyzer_dump_val(Empty3); // expected-warning {{lazyCompoundVal}}

  // We only have binding for the original Empty object, because copying empty
  // objects is a no-op in the performTrivialCopy. This is fine, because empty
  // objects don't have any data members that could be accessed anyway.
  clang_analyzer_printState();
  // CHECK:       "store": { "pointer": "0x{{[0-9a-f]+}}", "items": [
  // CHECK-NEXT:    { "cluster": "GlobalInternalSpaceRegion", "pointer": "0x{{[0-9a-f]+}}", "items": [
  // CHECK-NEXT:      { "kind": "Default", "offset": 0, "value": "conj_$
  // CHECK-NEXT:    ]},
  // CHECK-NEXT:    { "cluster": "GlobalSystemSpaceRegion", "pointer": "0x{{[0-9a-f]+}}", "items": [
  // CHECK-NEXT:      { "kind": "Default", "offset": 0, "value": "conj_$
  // CHECK-NEXT:    ]},
  // CHECK-NEXT:    { "cluster": "Empty", "pointer": "0x{{[0-9a-f]+}}", "items": [
  // CHECK-NEXT:      { "kind": "Default", "offset": 0, "value": "[[EMPTY_CONJ:conj_\$[0-9]+{int, LC[0-9]+, S[0-9]+, #[0-9]+}]]" }
  // CHECK-NEXT:    ]}
  // CHECK-NEXT:  ]},

  nop(Empty, Empty2, Empty3);
}

void _02_structs_with_members() {
  clang_analyzer_dump_val(conjure<aggr>()); // expected-warning {{lazyCompoundVal}}
  aggr Aggr = conjure<aggr>();
  aggr Aggr2 = Aggr;
  aggr Aggr3 = Aggr2;
  // All of these should refer to the exact same LCV, because all of
  // these trivial copies refer to the original conjured value.
  clang_analyzer_dump_val(Aggr);  // expected-warning {{lazyCompoundVal}}
  clang_analyzer_dump_val(Aggr2); // expected-warning {{lazyCompoundVal}}
  clang_analyzer_dump_val(Aggr3); // expected-warning {{lazyCompoundVal}}

  // We have fields in the struct we copy, thus we also have the entries for the copies
  // (and for all of their fields).
  clang_analyzer_printState();
  // CHECK:       "store": { "pointer": "0x{{[0-9a-f]+}}", "items": [
  // CHECK-NEXT:    { "cluster": "GlobalInternalSpaceRegion", "pointer": "0x{{[0-9a-f]+}}", "items": [
  // CHECK-NEXT:      { "kind": "Default", "offset": 0, "value": "conj_$
  // CHECK-NEXT:    ]},
  // CHECK-NEXT:    { "cluster": "GlobalSystemSpaceRegion", "pointer": "0x{{[0-9a-f]+}}", "items": [
  // CHECK-NEXT:      { "kind": "Default", "offset": 0, "value": "conj_$
  // CHECK-NEXT:    ]},
  // CHECK-NEXT:    { "cluster": "Aggr", "pointer": "0x{{[0-9a-f]+}}", "items": [
  // CHECK-NEXT:      { "kind": "Default", "offset": 0, "value": "[[AGGR_CONJ:conj_\$[0-9]+{int, LC[0-9]+, S[0-9]+, #[0-9]+}]]" }
  // CHECK-NEXT:    ]},
  // CHECK-NEXT:    { "cluster": "Aggr2", "pointer": "0x{{[0-9a-f]+}}", "items": [
  // CHECK-NEXT:      { "kind": "Direct", "offset": 0, "value": "derived_${{[0-9]+}}{[[AGGR_CONJ]],Aggr.x}" },
  // CHECK-NEXT:      { "kind": "Direct", "offset": 32, "value": "derived_${{[0-9]+}}{[[AGGR_CONJ]],Aggr.y}" }
  // CHECK-NEXT:    ]},
  // CHECK-NEXT:    { "cluster": "Aggr3", "pointer": "0x{{[0-9a-f]+}}", "items": [
  // CHECK-NEXT:      { "kind": "Direct", "offset": 0, "value": "derived_${{[0-9]+}}{[[AGGR_CONJ]],Aggr.x}" },
  // CHECK-NEXT:      { "kind": "Direct", "offset": 32, "value": "derived_${{[0-9]+}}{[[AGGR_CONJ]],Aggr.y}" }
  // CHECK-NEXT:    ]}
  // CHECK-NEXT:  ]},

  nop(Aggr, Aggr2, Aggr3);
}

// Tests that use `clang_analyzer_printState()` must share the analysis entry
// point, and have a strict ordering between. This is to meet the different
// `clang_analyzer_printState()` calls in a fixed relative ordering, thus
// FileCheck could check the stdouts.
void entrypoint() {
  _01_empty_structs();
  _02_structs_with_members();
}

} // namespace trivial_struct_copy

namespace gh153782 {

// Ensure we do not regress on the following use cases.
// The assumption made on a field in `setPtr` should apply to the returned copy in `func`.
struct Status { int error; };
Status getError();

Status setPtr(int **outptr, int* ptr) {
  Status e = getError();
  if (e.error != 0) return e; // When assuming the error field is non-zero,
  *outptr = ptr;              // this is not executed
  return e;
}

int func() {
  int *ptr = nullptr;
  int x = 42;
  if (setPtr(&ptr, &x).error == 0) {
    // The assumption made in get() SHOULD match the assumption about
    // the returned value, hence the engine SHOULD NOT assume ptr is null.
    clang_analyzer_dump_val(ptr); // expected-warning {{&x}}
    return *ptr;
  }
  return 0;
}

} // namespace gh153782
