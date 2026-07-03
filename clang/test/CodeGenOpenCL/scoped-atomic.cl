// RUN: %clang_cc1 %s -cl-std=CL2.0 -emit-llvm -o - -triple spir-unknown-unknown -verify
// RUN: %clang_cc1 %s -cl-std=CL2.0 -emit-llvm -o - -triple spir64-unknown-unknown -verify

// expected-no-diagnostics

int fi1a(int *i) {
  int v;
  __scoped_atomic_load(i, &v, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
  return v;
}

#ifdef __SPIR64__
long fl1a(long *i) {
  long v;
  __scoped_atomic_load(i, &v, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
  return v;
}
#endif
