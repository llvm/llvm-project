// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64-unknown-linux-gnu %s
// expected-no-diagnostics

// __int256 is never lock-free (256 bits > max atomic width on any current target)
_Static_assert(!__atomic_always_lock_free(32, 0), "__int256 should not be always lock-free");

// _Atomic __int256_t variables should compile
_Atomic __int256_t atomic_s256;
_Atomic __uint256_t atomic_u256;

// Atomic load/store should compile (will use libcalls)
__int256_t load_atomic(void) {
  return __c11_atomic_load(&atomic_s256, __ATOMIC_SEQ_CST);
}

void store_atomic(__int256_t val) {
  __c11_atomic_store(&atomic_s256, val, __ATOMIC_SEQ_CST);
}

__uint256_t load_atomic_unsigned(void) {
  return __c11_atomic_load(&atomic_u256, __ATOMIC_SEQ_CST);
}

void store_atomic_unsigned(__uint256_t val) {
  __c11_atomic_store(&atomic_u256, val, __ATOMIC_SEQ_CST);
}

// Atomic exchange
__int256_t exchange_atomic(__int256_t val) {
  return __c11_atomic_exchange(&atomic_s256, val, __ATOMIC_SEQ_CST);
}

__uint256_t exchange_atomic_unsigned(__uint256_t val) {
  return __c11_atomic_exchange(&atomic_u256, val, __ATOMIC_RELAXED);
}

// Atomic compare-exchange (strong and weak)
_Bool cas_strong(__int256_t *expected, __int256_t desired) {
  return __c11_atomic_compare_exchange_strong(
      &atomic_s256, expected, desired, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
}

_Bool cas_weak(__int256_t *expected, __int256_t desired) {
  return __c11_atomic_compare_exchange_weak(
      &atomic_s256, expected, desired, __ATOMIC_RELEASE, __ATOMIC_RELAXED);
}

_Bool cas_strong_unsigned(__uint256_t *expected, __uint256_t desired) {
  return __c11_atomic_compare_exchange_strong(
      &atomic_u256, expected, desired, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
}

_Bool cas_weak_unsigned(__uint256_t *expected, __uint256_t desired) {
  return __c11_atomic_compare_exchange_weak(
      &atomic_u256, expected, desired, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
}

// Different memory orderings for load/store
__int256_t load_relaxed(void) {
  return __c11_atomic_load(&atomic_s256, __ATOMIC_RELAXED);
}

__int256_t load_acquire(void) {
  return __c11_atomic_load(&atomic_s256, __ATOMIC_ACQUIRE);
}

void store_relaxed(__int256_t val) {
  __c11_atomic_store(&atomic_s256, val, __ATOMIC_RELAXED);
}

void store_release(__int256_t val) {
  __c11_atomic_store(&atomic_s256, val, __ATOMIC_RELEASE);
}
