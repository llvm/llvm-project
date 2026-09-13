// RUN: %check_clang_tidy -std=c17-or-earlier %s readability-non-const-parameter %t

static int f();

int f(p)
  int *p;
// CHECK-MESSAGES: :[[@LINE-1]]:8: warning: pointer parameter 'p' can be pointer to const [readability-non-const-parameter]
// CHECK-FIXES: const int *p;
{
    return *p;
}

// The 'expected' operand of a compare-exchange receives the old value when the
// exchange fails.
int atomic_cas(_Atomic int *obj, int *expected, int desired) {
  return __c11_atomic_compare_exchange_strong(
      obj, expected, desired, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
}

int atomic_cas_weak(_Atomic int *obj, int *expected, int desired) {
  return __c11_atomic_compare_exchange_weak(
      obj, expected, desired, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
}

// The non-'_n' load and exchange forms write their result through an operand.
void atomic_load_out(int *obj, int *dest) {
  __atomic_load(obj, dest, __ATOMIC_SEQ_CST);
}

void atomic_exchange_out(int *obj, int *val, int *old) {
  __atomic_exchange(obj, val, old, __ATOMIC_SEQ_CST);
}

// Only the operands of the atomic builtin are affected. Other parameters are
// still reported.
// CHECK-MESSAGES: :[[@LINE+1]]:69: warning: pointer parameter 'unrelated' can be pointer to const [readability-non-const-parameter]
int atomic_cas_unrelated(int *obj, int *expected, int desired, int *unrelated) {
  // CHECK-FIXES: int atomic_cas_unrelated(int *obj, int *expected, int desired, const int *unrelated) {
  return __atomic_compare_exchange_n(obj, expected, desired, 0,
                                     __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST) +
         *unrelated;
}

// Conservatively, operands of other atomic builtins are not suggested as const
// either, matching the existing treatment of ordinary function calls.
int atomic_load_ptr(int *p) {
  return __atomic_load_n(p, __ATOMIC_SEQ_CST);
}
