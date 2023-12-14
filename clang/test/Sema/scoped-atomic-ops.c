// RUN: %clang_cc1 -x c -triple=amdgcn-amd-amdhsa -verify -fsyntax-only %s
// RUN: %clang_cc1 -x c -triple=x86_64-pc-linux-gnu -verify -fsyntax-only %s

int fi1a(int *i) {
  int v;
  __scoped_atomic_load(i, &v, __ATOMIC_RELAXED); // expected-error {{too few arguments to function call, expected 4, have 3}}
  __scoped_atomic_load(i, &v, __ATOMIC_RELAXED, 42); // expected-error {{synchronization scope argument to atomic operation is invalid}}
  __scoped_atomic_load(i, &v, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  return v;
}

int fi1b(int *i) {
  *i = __scoped_atomic_load_n(i, __ATOMIC_RELAXED); // expected-error {{too few arguments to function call, expected 3, have 2}}
  *i = __scoped_atomic_load_n(i, __ATOMIC_RELAXED, 42); // expected-error {{synchronization scope argument to atomic operation is invalid}}
  *i = __scoped_atomic_load_n(i, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  return *i;
}

int fi2a(int *i) {
  int v;
  __scoped_atomic_store(i, &v, __ATOMIC_RELAXED); // expected-error {{too few arguments to function call, expected 4, have 3}}
  __scoped_atomic_store(i, &v, __ATOMIC_RELAXED, 42); // expected-error {{synchronization scope argument to atomic operation is invalid}}
  __scoped_atomic_store(i, &v, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  return v;
}

void fi2b(int *i) {
  __scoped_atomic_store_n(i, 1, __ATOMIC_RELAXED); // expected-error {{too few arguments to function call, expected 4, have 3}}
  __scoped_atomic_store_n(i, 1, __ATOMIC_RELAXED, 42); // expected-error {{synchronization scope argument to atomic operation is invalid}}
  __scoped_atomic_store_n(i, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
}

void fi3a(int *a, int *b, int *c, int *d, int *e, int *f, int *g, int *h) {
  *a = __scoped_atomic_fetch_add(a, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  *b = __scoped_atomic_fetch_sub(b, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  *c = __scoped_atomic_fetch_and(c, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  *d = __scoped_atomic_fetch_or(d, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  *e = __scoped_atomic_fetch_xor(e, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  *f = __scoped_atomic_fetch_nand(f, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  *g = __scoped_atomic_fetch_min(g, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  *h = __scoped_atomic_fetch_max(h, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
}

void fi3b(int *a, int *b, int *c, int *d, int *e, int *f, int *g, int *h) {
  *a = __scoped_atomic_fetch_add(1, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM); // expected-error {{address argument to atomic builtin must be a pointer ('int' invalid)}}
  *b = __scoped_atomic_fetch_sub(1, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM); // expected-error {{address argument to atomic builtin must be a pointer ('int' invalid)}}
  *c = __scoped_atomic_fetch_and(1, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM); // expected-error {{address argument to atomic builtin must be a pointer ('int' invalid)}}
  *d = __scoped_atomic_fetch_or(1, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM); // expected-error {{address argument to atomic builtin must be a pointer ('int' invalid)}}
  *e = __scoped_atomic_fetch_xor(1, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM); // expected-error {{address argument to atomic builtin must be a pointer ('int' invalid)}}
  *f = __scoped_atomic_fetch_nand(1, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM); // expected-error {{address argument to atomic builtin must be a pointer ('int' invalid)}}
  *g = __scoped_atomic_fetch_min(1, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM); // expected-error {{address argument to atomic builtin must be a pointer ('int' invalid)}}
  *h = __scoped_atomic_fetch_max(1, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM); // expected-error {{address argument to atomic builtin must be a pointer ('int' invalid)}}
}

void fi3c(int *a, int *b, int *c, int *d, int *e, int *f, int *g, int *h) {
  *a = __scoped_atomic_fetch_add(a, 1, __ATOMIC_RELAXED); // expected-error {{too few arguments to function call, expected 4, have 3}}
  *b = __scoped_atomic_fetch_sub(b, 1, __ATOMIC_RELAXED); // expected-error {{too few arguments to function call, expected 4, have 3}}
  *c = __scoped_atomic_fetch_and(c, 1, __ATOMIC_RELAXED); // expected-error {{too few arguments to function call, expected 4, have 3}}
  *d = __scoped_atomic_fetch_or(d, 1, __ATOMIC_RELAXED); // expected-error {{too few arguments to function call, expected 4, have 3}}
  *e = __scoped_atomic_fetch_xor(e, 1, __ATOMIC_RELAXED); // expected-error {{too few arguments to function call, expected 4, have 3}}
  *f = __scoped_atomic_fetch_nand(f, 1, __ATOMIC_RELAXED); // expected-error {{too few arguments to function call, expected 4, have 3}}
  *g = __scoped_atomic_fetch_min(g, 1, __ATOMIC_RELAXED); // expected-error {{too few arguments to function call, expected 4, have 3}}
  *h = __scoped_atomic_fetch_max(h, 1, __ATOMIC_RELAXED); // expected-error {{too few arguments to function call, expected 4, have 3}}
}

void fi3d(int *a, int *b, int *c, int *d, int *e, int *f, int *g, int *h) {
  *a = __scoped_atomic_fetch_add(a, 1, __ATOMIC_RELAXED, 42); // expected-error {{synchronization scope argument to atomic operation is invalid}}
  *b = __scoped_atomic_fetch_sub(b, 1, __ATOMIC_RELAXED, 42); // expected-error {{synchronization scope argument to atomic operation is invalid}}
  *c = __scoped_atomic_fetch_and(c, 1, __ATOMIC_RELAXED, 42); // expected-error {{synchronization scope argument to atomic operation is invalid}}
  *d = __scoped_atomic_fetch_or(d, 1, __ATOMIC_RELAXED, 42); // expected-error {{synchronization scope argument to atomic operation is invalid}}
  *e = __scoped_atomic_fetch_xor(e, 1, __ATOMIC_RELAXED, 42); // expected-error {{synchronization scope argument to atomic operation is invalid}}
  *f = __scoped_atomic_fetch_nand(f, 1, __ATOMIC_RELAXED, 42); // expected-error {{synchronization scope argument to atomic operation is invalid}}
  *g = __scoped_atomic_fetch_min(g, 1, __ATOMIC_RELAXED, 42); // expected-error {{synchronization scope argument to atomic operation is invalid}}
  *h = __scoped_atomic_fetch_max(h, 1, __ATOMIC_RELAXED, 42); // expected-error {{synchronization scope argument to atomic operation is invalid}}
}

int fi4a(int *i) {
  int cmp = 0;
  int desired = 1;
  return __scoped_atomic_compare_exchange(i, &cmp, &desired, 0,
                                          __ATOMIC_ACQUIRE, __ATOMIC_ACQUIRE,
                                          __MEMORY_SCOPE_SYSTEM);
}

int fi5a(int *i) {
  int cmp = 0;
  return __scoped_atomic_compare_exchange_n(i, &cmp, 1, 1, __ATOMIC_ACQUIRE,
                                            __ATOMIC_ACQUIRE,
                                            __MEMORY_SCOPE_SYSTEM);
}

int fi6a(int *c, int *d) {
  int ret;
  __scoped_atomic_exchange(c, d, &ret, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  return ret;
}

int fi7a(_Bool *c) {
  return __scoped_atomic_exchange_n(c, 1, __ATOMIC_RELAXED,
                                    __MEMORY_SCOPE_SYSTEM);
}
