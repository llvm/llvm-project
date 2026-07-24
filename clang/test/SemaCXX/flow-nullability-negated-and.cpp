// Regression: `if (x && !(a && b))` must narrow 'x' on the true edge only.
// The negation path decomposes the nested `!(a && b)` and flips its leaves to
// the false edge; doing that on the shared Results vector clobbered the outer
// decomposeAnd's already-appended 'x' leaf, so 'x' lost true-edge narrowing
// (false positive on the dereference) and gained false-edge narrowing (false
// negative). The fix decomposes the nested `&&` into a local vector.
//
// RUN: %clang_cc1 -fsyntax-only -fflow-sensitive-nullability -fnullability-default=nullable -Wno-nullable-to-nonnull-conversion -std=c++17 -Wno-unused-value %s -verify

void negated_nested_and(int *_Nullable x, int *_Nullable a, int *_Nullable b) {
  if (x && !(a && b)) {
    (void)*x; // true edge: x proven non-null, no warning
  } else {
    // False edge is reached when x is null (or a&&b is true with x non-null),
    // so x is NOT guaranteed non-null here and the dereference must warn.
    (void)*x; // expected-warning{{dereference of nullable pointer}} expected-note{{add a null check}}
  }
}

// Sanity: the plain (un-negated) outer-&&-with-nested-&& still narrows all of
// x, a, b on the true edge.
void plain_nested_and(int *_Nullable x, int *_Nullable a, int *_Nullable b) {
  if (x && (a && b)) {
    (void)*x; // no warning
    (void)*a; // no warning
    (void)*b; // no warning
  }
}
