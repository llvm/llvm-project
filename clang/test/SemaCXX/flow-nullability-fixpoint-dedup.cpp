// Regression: diagnostics must be emitted once after the dataflow fixpoint
// converges, not from inside the transfer functions. A block re-visited with a
// changed entry state (because some pointer's narrowing flips across a loop
// back-edge) re-runs its transfer functions and would re-report the same
// nullable dereference. Before the collect-then-emit fix this surfaced as
// duplicate warnings; -verify here pins the count to exactly one per location.
//
// RUN: %clang_cc1 -fsyntax-only -fflow-sensitive-nullability -fnullability-default=nullable -Wno-nullable-to-nonnull-conversion -std=c++17 -Wno-unused-value %s -verify

extern bool cond();

// 'r' is narrowed entering the loop but becomes nullable at the end of the
// body, so the back-edge changes the loop header's merged entry state and the
// body is processed a second time. 'p' is never narrowed, so its dereference
// fires the transfer-function callback on every visit. The buffered, deduped
// reporter must still produce exactly one warning.
void dup_deref_across_backedge(int *_Nullable p, int *_Nullable r) {
  if (!r)
    return; // r non-null below
  while (cond()) {
    (void)*p; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
    r = p;    // r becomes nullable; flows back to the header, forcing a re-visit
    (void)r;
  }
}

// A second nullable pointer in the same re-visited block, to confirm the dedup
// key distinguishes locations (each still reports exactly once, not N times).
void two_derefs_one_each(int *_Nullable p, int *_Nullable q, int *_Nullable r) {
  if (!r)
    return;
  while (cond()) {
    (void)*p; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
    (void)*q; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
    r = p;
    (void)r;
  }
}
