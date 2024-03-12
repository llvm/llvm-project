// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -verify=expected,both -Wno-unused-value %s
// RUN: %clang_cc1 -verify=ref,both -Wno-unused-value %s

void blah() {
  __complex__ unsigned xx;
  __complex__ signed yy;
  __complex__ int result;

  /// The following line calls into the constant interpreter.
  result = xx * yy;
}


_Static_assert((0.0 + 0.0j) == (0.0 + 0.0j), "");
_Static_assert((0.0 + 0.0j) != (0.0 + 0.0j), ""); // both-error {{static assertion}} \
                                                  // both-note {{evaluates to}}

const _Complex float FC = {0.0f, 0.0f};
_Static_assert(!FC, "");
const _Complex float FI = {0, 0};
_Static_assert(!FI, "");
