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
