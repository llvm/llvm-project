// RUN: not %clang_cc1 -std=c++20 -fsyntax-only %s 2>&1 \
// RUN:   | FileCheck %s --implicit-check-not='Assertion `NumElements.isPositive()` failed'

// In C++20, constexpr dynamic allocation is permitted only if valid.
// A negative element count must be diagnosed (and must not crash).

constexpr void f_bad_neg() {
  int a = -1;
  (void) new int[a]; // triggers negative-size path in the interpreter
}

struct __nothrow_t { };
extern const __nothrow_t __nothrow_dummy;
void* operator new[](unsigned long, const __nothrow_t&) noexcept;

// Ensure we take the nothrow overload.
constexpr void f_bad_neg_nothrow() {
  (void) new (__nothrow_dummy) int[-7]; // should evaluate to nullptr (no crash)
}

// Force evaluation so the constexpr interpreter actually runs both cases.
constexpr bool force_eval1 = (f_bad_neg(), true);
constexpr bool force_eval2 = (f_bad_neg_nothrow(), true);

// CHECK: error: constexpr function {{(never produces|is not a)}} constant expression