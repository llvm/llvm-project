// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64-unknown-linux-gnu %s

// Verify that __sync_* builtins reject __int256 (max atomic width is 16 bytes).
// The __c11_atomic_* builtins accept __int256 (via libcalls) and are tested
// separately in atomic-int256.c and CodeGen/X86/x86_64-atomic-i256.c.

__int256 test_sync_add(__int256 *addr, __int256 val) {
  return __sync_fetch_and_add(addr, val); // expected-error {{address argument to atomic builtin must be a pointer to 1,2,4,8 or 16 byte type}}
}

__int256 test_sync_sub(__int256 *addr, __int256 val) {
  return __sync_fetch_and_sub(addr, val); // expected-error {{address argument to atomic builtin must be a pointer to 1,2,4,8 or 16 byte type}}
}

__int256 test_sync_or(__int256 *addr, __int256 val) {
  return __sync_fetch_and_or(addr, val); // expected-error {{address argument to atomic builtin must be a pointer to 1,2,4,8 or 16 byte type}}
}

__int256 test_sync_and(__int256 *addr, __int256 val) {
  return __sync_fetch_and_and(addr, val); // expected-error {{address argument to atomic builtin must be a pointer to 1,2,4,8 or 16 byte type}}
}

__int256 test_sync_xor(__int256 *addr, __int256 val) {
  return __sync_fetch_and_xor(addr, val); // expected-error {{address argument to atomic builtin must be a pointer to 1,2,4,8 or 16 byte type}}
}

_Bool test_sync_cas(__int256 *addr, __int256 oldval, __int256 newval) {
  return __sync_bool_compare_and_swap(addr, oldval, newval); // expected-error {{address argument to atomic builtin must be a pointer to 1,2,4,8 or 16 byte type}}
}
