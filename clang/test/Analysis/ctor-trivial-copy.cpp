// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -analyzer-config c++-inlining=constructors -verify %s \
// RUN:   2>&1 | FileCheck %s


void clang_analyzer_printState();
template <typename T> void clang_analyzer_dump_lref(T& param);
template <typename T> void clang_analyzer_dump_val(T param);
template <typename T> void clang_analyzer_denote(T param, const char *name);
template <typename T> void clang_analyzer_express(T param);
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
  clang_analyzer_dump_val(conjure<empty>()); // expected-warning {{conj_$}}
  empty Empty = conjure<empty>();
  empty Empty2 = Empty;
  empty Empty3 = Empty2;

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
  clang_analyzer_dump_val(conjure<aggr>()); // expected-warning {{conj_$}}
  aggr Aggr = conjure<aggr>();
  aggr Aggr2 = Aggr;
  aggr Aggr3 = Aggr2;
  // All of these should refer to the exact same symbol, because all of
  // these trivial copies refer to the original conjured value.
  clang_analyzer_denote(Aggr, "$Aggr");
  clang_analyzer_express(Aggr);  // expected-warning {{$Aggr}}
  clang_analyzer_express(Aggr2); // expected-warning {{$Aggr}}
  clang_analyzer_express(Aggr3); // expected-warning {{$Aggr}}

  // We should have the same Conjured symbol for "Aggr", "Aggr2" and "Aggr3".
  // We used to have Derived symbols for the individual fields that were
  // copied as part of copying the whole struct.
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
  // CHECK-NEXT:      { "kind": "Default", "offset": 0, "value": "[[AGGR_CONJ]]" }
  // CHECK-NEXT:    ]},
  // CHECK-NEXT:    { "cluster": "Aggr3", "pointer": "0x{{[0-9a-f]+}}", "items": [
  // CHECK-NEXT:      { "kind": "Default", "offset": 0, "value": "[[AGGR_CONJ]]" }
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
