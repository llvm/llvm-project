// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify %s

// Verify that values produced by ProgramState::invalidateRegions during a
// conservative-eval call are bound as SymbolInvalidationArtifact symbols
// (printed as "inv_$N{... cause, #...}"). Symbols produced by other paths
// (e.g. uninvalidated initial loads) keep their existing kinds.

void clang_analyzer_dump(int);
void clang_analyzer_dump_ptr(int *);
void clang_analyzer_eval(int);

int gGlobal;
void opaque(int *);

void test_conservative_call_invalidates_arg(void) {
  int x = 0;
  opaque(&x);
  // Scalar conjure site (RegionStore.cpp BindingKey::Direct path).
  clang_analyzer_dump(x); // expected-warning-re{{{{inv_\$[0-9]+{int, LC[0-9]+, conservative-call, S[0-9]+, #1}}}}}

  // Constraint manager dispatch still works on the new kind.
  if (x == 42) {
    clang_analyzer_eval(x == 42); // expected-warning {{TRUE}}
  }
}

void test_conservative_call_invalidates_globals(void) {
  gGlobal = 0;
  opaque((int *)0);
  // Global memory-space conjure site (RegionStore::invalidateGlobalRegion).
  clang_analyzer_dump(gGlobal); // expected-warning-re{{{{derived_\$[0-9]+{inv_\$[0-9]+{int, LC[0-9]+, conservative-call, S[0-9]+, #1},gGlobal}}}}}
}

struct Foo { int x; int y; };
void opaqueStruct(struct Foo *);

void test_conservative_call_invalidates_record(void) {
  struct Foo s = {0, 0};
  opaqueStruct(&s);
  // Record-type conjure site (RegionStore.cpp record default-binding path):
  // members are read as derived_$ over an inv_$ default binding.
  clang_analyzer_dump(s.x); // expected-warning-re{{{{derived_\$[0-9]+{inv_\$[0-9]+{int, LC[0-9]+, conservative-call, S[0-9]+, #1},s.x}}}}}
}

void opaqueArr(int *);

void test_conservative_call_invalidates_array(void) {
  int arr[3] = {0, 0, 0};
  opaqueArr(arr);
  // Array-element conjure site (RegionStore.cpp array default-binding path).
  clang_analyzer_dump(arr[0]); // expected-warning-re{{{{derived_\$[0-9]+{inv_\$[0-9]+{int, LC[0-9]+, conservative-call, S[0-9]+, #1},Element{arr,0 S64b,int}}}}}}
}

int *opaqueHeap(void);
void opaquePtr(int *);

void test_conservative_call_invalidates_symbolic_region(void) {
  int *p = opaqueHeap();
  opaquePtr(p);
  // SymbolicRegion conjure site (RegionStore.cpp alloca/symbolic path):
  // dereferencing the heap pointer after the opaque call yields a value
  // derived from an inv_$ default binding on the symbolic region.
  clang_analyzer_dump(*p); // expected-warning-re{{{{inv_\$[0-9]+{int, LC[0-9]+, conservative-call, S[0-9]+, #1}}}}}
}

int returnsInt(void);

void test_eval_call_returns_conjured(void) {
  // A pure return value from an opaque call is a SymbolConjured (no
  // invalidation event) — ensure we did not regress that path.
  int r = returnsInt();
  clang_analyzer_dump(r); // expected-warning-re{{{{conj_\$[0-9]+{int, LC[0-9]+, S[0-9]+, #1}}}}}
}

void test_previous_symbol_is_recorded(void) {
  // Bind x to a conjured symbol first, then invalidate it. The resulting
  // SymbolInvalidationArtifact must carry that prior symbol via
  // getPreviousSymbol(); the dump surfaces it as "prev=conj_$...".
  int x = returnsInt();
  opaque(&x);
  clang_analyzer_dump(x); // expected-warning-re{{{{inv_\$[0-9]+{int, LC[0-9]+, conservative-call, S[0-9]+, prev=conj_\$[0-9]+{int, LC[0-9]+, S[0-9]+, #1}, #1}}}}}
}
