// RUN: %clang_cc1 -x c -verify -fsyntax-only %s
// RUN: %clang_cc1 -x c -Watomic-memory-scope -verify=enabled -fsyntax-only %s

// Test warnings for scope arguments:
// - Integer arguments trigger warning only when -Watomic-memory-scope is enabled
// - Wrong enum types always trigger warning

void test_integer_literal_scope(int *ptr) {
  int val;
  // Using integer literals: warning only with -Watomic-memory-scope
  __scoped_atomic_load(ptr, &val, __ATOMIC_RELAXED, 0); // enabled-warning {{synchronization scope should be of type __memory_scope}}
  __scoped_atomic_load(ptr, &val, __ATOMIC_RELAXED, 1); // enabled-warning {{synchronization scope should be of type __memory_scope}}
  *ptr = __scoped_atomic_load_n(ptr, __ATOMIC_RELAXED, 2); // enabled-warning {{synchronization scope should be of type __memory_scope}}
}

void test_integer_variable_scope(int *ptr) {
  int val;
  int scope = 0;
  // Using integer variables: warning only with -Watomic-memory-scope
  __scoped_atomic_load(ptr, &val, __ATOMIC_RELAXED, scope); // enabled-warning {{synchronization scope should be of type __memory_scope}}
  *ptr = __scoped_atomic_load_n(ptr, __ATOMIC_RELAXED, scope); // enabled-warning {{synchronization scope should be of type __memory_scope}}
}

// Test warning for wrong enum type
enum my_scope {
  MY_SCOPE_SYSTEM = 0,
  MY_SCOPE_DEVICE = 1
};

void test_wrong_enum_type(int *ptr) {
  int val;
  // Using wrong enum type should trigger warning (enabled by default)
  // Test with enumerators (in C these have type int, triggering DeclRefExpr check)
  __scoped_atomic_load(ptr, &val, __ATOMIC_RELAXED, MY_SCOPE_SYSTEM); // expected-warning {{synchronization scope should be of type __memory_scope}} enabled-warning {{synchronization scope should be of type __memory_scope}}
  *ptr = __scoped_atomic_load_n(ptr, __ATOMIC_RELAXED, MY_SCOPE_DEVICE); // expected-warning {{synchronization scope should be of type __memory_scope}} enabled-warning {{synchronization scope should be of type __memory_scope}}
  __scoped_atomic_thread_fence(__ATOMIC_SEQ_CST, MY_SCOPE_SYSTEM); // expected-warning {{synchronization scope should be of type __memory_scope}} enabled-warning {{synchronization scope should be of type __memory_scope}}

  // Test with variable of wrong enum type (triggers enum type check)
  enum my_scope wrong_scope = MY_SCOPE_SYSTEM;
  __scoped_atomic_load(ptr, &val, __ATOMIC_RELAXED, wrong_scope); // expected-warning {{synchronization scope should be of type __memory_scope}} enabled-warning {{synchronization scope should be of type __memory_scope}}
}

void test_various_scoped_atomics_with_integer(int *ptr) {
  // Test warnings with various scoped atomic operations
  *ptr = __scoped_atomic_fetch_add(ptr, 1, __ATOMIC_RELAXED, 0); // enabled-warning {{synchronization scope should be of type __memory_scope}}
  *ptr = __scoped_atomic_add_fetch(ptr, 1, __ATOMIC_RELAXED, 1); // enabled-warning {{synchronization scope should be of type __memory_scope}}
  *ptr = __scoped_atomic_fetch_sub(ptr, 1, __ATOMIC_RELAXED, 2); // enabled-warning {{synchronization scope should be of type __memory_scope}}
  __scoped_atomic_store_n(ptr, 1, __ATOMIC_RELAXED, 3); // enabled-warning {{synchronization scope should be of type __memory_scope}}
}

void test_fence_with_integer() {
  // Test warnings with __scoped_atomic_thread_fence using integer literals
  __scoped_atomic_thread_fence(__ATOMIC_SEQ_CST, 0); // enabled-warning {{synchronization scope should be of type __memory_scope}}
  __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE, 1); // enabled-warning {{synchronization scope should be of type __memory_scope}}

  int scope_var = 2;
  __scoped_atomic_thread_fence(__ATOMIC_RELEASE, scope_var); // enabled-warning {{synchronization scope should be of type __memory_scope}}
}
