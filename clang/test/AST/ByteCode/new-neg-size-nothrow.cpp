// RUN: %clang_cc1 -std=c++20 -fsyntax-only -fexperimental-new-constant-interpreter -verify %s
// expected-no-diagnostics

struct __nothrow_t { };
extern const __nothrow_t __nothrow_dummy;
void* operator new[](unsigned long, const __nothrow_t&) noexcept;

// This test ensures that new (nothrow) int[-1] does not crash in constexpr interpreter.
// It should evaluate to a nullptr, not assert.
constexpr int get_neg_size() {
  return -1;
}

void test_nothrow_negative_size() {
  int x = get_neg_size();
  int *p = new (__nothrow_dummy) int[x]; 
  (void)p;
}
