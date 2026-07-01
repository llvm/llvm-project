// Stage 2c: prove the redirect resolves correctly.
// Include BOTH modules so A loads first (registers shared.h as the canonical
// copy) and B loads second (its shared.h is de-duplicated / redirected into
// A's copy). Then trigger a diagnostic whose location lives in shared.h.
// The error must still point at shared.h:<line> with the right function/line,
// which only holds if the redirected offsets resolve correctly.
#include "a.h"
#include "b.h"

int main() {
  // shared_fn_1 is defined in shared.h (present in both A and B); pass a bad
  // argument so overload resolution reports the candidate in shared.h.
  return a_entry(1) + shared_fn_1("oops");
}
