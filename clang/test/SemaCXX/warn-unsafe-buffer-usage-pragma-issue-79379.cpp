// RUN: %clang_cc1 -std=c++20 -verify %s

// Test that -Wunsafe-buffer-usage respects #pragma clang diagnostic
// push/pop, just like other diagnostics such as -Wsign-compare.
//
// Previously, -Wunsafe-buffer-usage analysis ran at end-of-TU, which caused
// a nonlocal effect: the warning only fired if it was enabled at both the
// point of occurrence AND at the end of the file. With per-Decl analysis,
// the warning now correctly follows the diagnostic state at each function
// definition.

// No warning here: -Wunsafe-buffer-usage is not enabled.
void f1(int *x) { (void)x[1]; }

#pragma clang diagnostic push
#pragma clang diagnostic warning "-Wunsafe-buffer-usage"

// Warning here: -Wunsafe-buffer-usage is enabled by the pragma above.
void f2(int *x) { (void)x[1]; } // expected-warning{{unsafe buffer access}} \
                                    expected-note{{pass -fsafe-buffer-usage-suggestions to receive code hardening suggestions}}

#pragma clang diagnostic pop

// No warning here: the pop restored the previous state (disabled).
void f3(int *x) { (void)x[1]; }
