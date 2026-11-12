// RUN: %clang_analyze_cc1 -std=c++14 \
// RUN:   -analyzer-checker=core,cplusplus.Move,alpha.cplusplus.SmartPtr \
// RUN:   -analyzer-checker=cplusplus.NewDelete,cplusplus.NewDeleteLeaks \
// RUN:   -analyzer-config cplusplus.SmartPtrModeling:ModelSmartPtrDereference=true \
// RUN:   -verify %s

// Tests that SmartPtrModeling correctly hands off ownership to MallocChecker
// when unique_ptr::release() transfers a new-allocated pointer back to the
// caller. The inter-checker API (allocation_state::isReleasedByNew /
// transferToCallerNew) transitions the symbol from Escaped to Allocated in
// MallocChecker's RegionState, enabling cplusplus.NewDeleteLeaks to report
// a memory leak if the caller fails to delete the raw pointer.
//
// The handoff is gated on isReleasedByNew(), which requires MallocChecker to
// have prior knowledge of the symbol (AF_CXXNew + Escaped state). This ensures
// we never register unknown-provenance symbols (e.g., from unique_ptr parameters)
// and produce no false positives in that case.

#include "Inputs/system-header-simulator-cxx.h"

// ---------------------------------------------------------------------------
// Positive: leak must be reported
// ---------------------------------------------------------------------------

void leak_release_discarded() {
  std::unique_ptr<int> P(new int(42));
  P.release(); // raw pointer discarded — definite leak
} // expected-warning{{Potential memory leak [cplusplus.NewDeleteLeaks]}}

void leak_release_stored_not_freed() {
  std::unique_ptr<int> P(new int(42));
  int *Raw = P.release(); // caller takes ownership but never deletes
  (void)Raw;
} // expected-warning{{Potential leak of memory pointed to by 'Raw' [cplusplus.NewDeleteLeaks]}}

// ---------------------------------------------------------------------------
// Negative: no false positives
// ---------------------------------------------------------------------------

void no_leak_release_then_delete() {
  std::unique_ptr<int> P(new int(42));
  int *Raw = P.release();
  delete Raw; // correctly freed — no warning
}

void no_false_positive_parameter(std::unique_ptr<int> P) {
  // P has unknown provenance: MallocChecker never saw the new expression, so
  // isReleasedByNew() returns false. We conservatively skip the handoff to
  // avoid emitting spurious leak warnings on all callers of such functions.
  int *Raw = P.release();
  (void)Raw; // no warning
}

void no_leak_destructor_runs() {
  std::unique_ptr<int> P(new int(42));
  // release() not called: unique_ptr destructor frees the memory — no leak.
}
