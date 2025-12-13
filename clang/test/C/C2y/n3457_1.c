// RUN: %clang_cc1 -verify -std=c2y -finitial-counter-value=2147483646 %s

// The value produced needs to be a type that's representable with a signed
// long. However, the actual type it expands to does *not* need to be forced to
// be signed long because that would generally mean suffixing the value with L,
// which would be very surprising for folks using this to generate unique ids.
// We'll test this by ensuring the largest value can be expanded properly and
// an assertion that signed long is always at least four bytes wide (which is
// what's required to represent that maximal value).
//
// So we set the initial counter value to 2147483646, we'll validate that,
// increment it once to get to the maximal value and ensure there's no
// diagnostic, then increment again to ensure we get the constraint violation.

static_assert(__COUNTER__ == 2147483646); // Test and increment
static_assert(__COUNTER__ == 2147483647); // Test and increment

// This one should fail.
signed long i = __COUNTER__; // expected-error {{'__COUNTER__' value cannot exceed 2'147'483'647}}

