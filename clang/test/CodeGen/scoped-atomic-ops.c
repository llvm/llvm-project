// RUN: %clang_cc1 %s -emit-llvm -o - -triple=amdgcn-amd-amdhsa -ffreestanding \
// RUN:   -fvisibility=hidden | FileCheck %s

// CHECK-LABEL: define hidden i32 @fi1a(
// CHECK:    [[TMP0:%.*]] = load atomic i32, ptr [[PTR0:.+]] syncscope("one-as") monotonic, align 4
// CHECK:    [[TMP1:%.*]] = load atomic i32, ptr [[PTR1:.+]] syncscope("agent-one-as") monotonic, align 4
// CHECK:    [[TMP2:%.*]] = load atomic i32, ptr [[PTR2:.+]] syncscope("workgroup-one-as") monotonic, align 4
// CHECK:    [[TMP3:%.*]] = load atomic i32, ptr [[PTR3:.+]] syncscope("wavefront-one-as") monotonic, align 4
// CHECK:    [[TMP4:%.*]] = load atomic i32, ptr [[PTR4:.+]] syncscope("singlethread-one-as") monotonic, align 4
int fi1a(int *i) {
  int v;
  __scoped_atomic_load(i, &v, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  __scoped_atomic_load(i, &v, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
  __scoped_atomic_load(i, &v, __ATOMIC_RELAXED, __MEMORY_SCOPE_WRKGRP);
  __scoped_atomic_load(i, &v, __ATOMIC_RELAXED, __MEMORY_SCOPE_WVFRNT);
  __scoped_atomic_load(i, &v, __ATOMIC_RELAXED, __MEMORY_SCOPE_SINGLE);
  return v;
}

// CHECK-LABEL: define hidden i32 @fi1b(
// CHECK:    [[TMP0:%.*]] = load atomic i32, ptr [[PTR0:%.+]] syncscope("one-as") monotonic, align 4
// CHECK:    [[TMP1:%.*]] = load atomic i32, ptr [[PTR1:%.+]] syncscope("agent-one-as") monotonic, align 4
// CHECK:    [[TMP2:%.*]] = load atomic i32, ptr [[PTR2:%.+]] syncscope("workgroup-one-as") monotonic, align 4
// CHECK:    [[TMP3:%.*]] = load atomic i32, ptr [[PTR3:%.+]] syncscope("wavefront-one-as") monotonic, align 4
// CHECK:    [[TMP4:%.*]] = load atomic i32, ptr [[PTR4:%.+]] syncscope("singlethread-one-as") monotonic, align 4
//
int fi1b(int *i) {
  *i = __scoped_atomic_load_n(i, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  *i = __scoped_atomic_load_n(i, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
  *i = __scoped_atomic_load_n(i, __ATOMIC_RELAXED, __MEMORY_SCOPE_WRKGRP);
  *i = __scoped_atomic_load_n(i, __ATOMIC_RELAXED, __MEMORY_SCOPE_WVFRNT);
  *i = __scoped_atomic_load_n(i, __ATOMIC_RELAXED, __MEMORY_SCOPE_SINGLE);
  return *i;
}

// CHECK-LABEL: define hidden void @fi2a(
// CHECK:    store atomic i32 [[TMP0:%.+]], ptr [[PTR0:%.+]] syncscope("one-as") monotonic, align 4
// CHECK:    store atomic i32 [[TMP1:%.+]], ptr [[PTR1:%.+]] syncscope("agent-one-as") monotonic, align 4
// CHECK:    store atomic i32 [[TMP2:%.+]], ptr [[PTR2:%.+]] syncscope("workgroup-one-as") monotonic, align 4
// CHECK:    store atomic i32 [[TMP3:%.+]], ptr [[PTR3:%.+]] syncscope("wavefront-one-as") monotonic, align 4
// CHECK:    store atomic i32 [[TMP4:%.+]], ptr [[PTR4:%.+]] syncscope("singlethread-one-as") monotonic, align 4
//
void fi2a(int *i) {
  int v = 1;
  __scoped_atomic_store(i, &v, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  __scoped_atomic_store(i, &v, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
  __scoped_atomic_store(i, &v, __ATOMIC_RELAXED, __MEMORY_SCOPE_WRKGRP);
  __scoped_atomic_store(i, &v, __ATOMIC_RELAXED, __MEMORY_SCOPE_WVFRNT);
  __scoped_atomic_store(i, &v, __ATOMIC_RELAXED, __MEMORY_SCOPE_SINGLE);
}

// CHECK-LABEL: define hidden void @fi2b(
// CHECK:    store atomic i32 [[TMP0:%.+]], ptr [[PTR0:%.+]] syncscope("one-as") monotonic, align 4
// CHECK:    store atomic i32 [[TMP1:%.+]], ptr [[PTR1:%.+]] syncscope("agent-one-as") monotonic, align 4
// CHECK:    store atomic i32 [[TMP2:%.+]], ptr [[PTR2:%.+]] syncscope("workgroup-one-as") monotonic, align 4
// CHECK:    store atomic i32 [[TMP3:%.+]], ptr [[PTR3:%.+]] syncscope("wavefront-one-as") monotonic, align 4
// CHECK:    store atomic i32 [[TMP4:%.+]], ptr [[PTR4:%.+]] syncscope("singlethread-one-as") monotonic, align 4
void fi2b(int *i) {
  __scoped_atomic_store_n(i, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  __scoped_atomic_store_n(i, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
  __scoped_atomic_store_n(i, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_WRKGRP);
  __scoped_atomic_store_n(i, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_WVFRNT);
  __scoped_atomic_store_n(i, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SINGLE);
}

// CHECK-LABEL: define hidden void @fi3a(
// CHECK:    [[TMP0:%.*]] = atomicrmw add ptr [[PTR0:%.+]], i32 [[VAL0:.+]] syncscope("one-as") monotonic, align 4
// CHECK:    [[TMP1:%.*]] = atomicrmw sub ptr [[PTR1:%.+]], i32 [[VAL1:.+]] syncscope("one-as") monotonic, align 4
// CHECK:    [[TMP2:%.*]] = atomicrmw and ptr [[PTR2:%.+]], i32 [[VAL2:.+]] syncscope("one-as") monotonic, align 4
// CHECK:    [[TMP3:%.*]] = atomicrmw or ptr [[PTR3:%.+]], i32 [[VAL3:.+]] syncscope("one-as") monotonic, align 4
// CHECK:    [[TMP4:%.*]] = atomicrmw xor ptr [[PTR4:%.+]], i32 [[VAL4:.+]] syncscope("one-as") monotonic, align 4
// CHECK:    [[TMP5:%.*]] = atomicrmw nand ptr [[PTR5:%.+]], i32 [[VAL5:.+]] syncscope("one-as") monotonic, align 4
// CHECK:    [[TMP6:%.*]] = atomicrmw min ptr [[PTR6:%.+]], i32 [[VAL6:.+]] syncscope("one-as") monotonic, align 4
// CHECK:    [[TMP7:%.*]] = atomicrmw max ptr [[PTR7:%.+]], i32 [[VAL7:.+]] syncscope("one-as") monotonic, align 4
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

// CHECK-LABEL: define hidden void @fi3b(
// CHECK:    [[TMP0:%.*]] = atomicrmw add ptr [[PTR0:%.+]], i32 [[VAL0:.+]] syncscope("agent-one-as") monotonic, align 4
// CHECK:    [[TMP1:%.*]] = atomicrmw sub ptr [[PTR1:%.+]], i32 [[VAL1:.+]] syncscope("agent-one-as") monotonic, align 4
// CHECK:    [[TMP2:%.*]] = atomicrmw and ptr [[PTR2:%.+]], i32 [[VAL2:.+]] syncscope("agent-one-as") monotonic, align 4
// CHECK:    [[TMP3:%.*]] = atomicrmw or ptr [[PTR3:%.+]], i32 [[VAL3:.+]] syncscope("agent-one-as") monotonic, align 4
// CHECK:    [[TMP4:%.*]] = atomicrmw xor ptr [[PTR4:%.+]], i32 [[VAL4:.+]] syncscope("agent-one-as") monotonic, align 4
// CHECK:    [[TMP5:%.*]] = atomicrmw nand ptr [[PTR5:%.+]], i32 [[VAL5:.+]] syncscope("agent-one-as") monotonic, align 4
// CHECK:    [[TMP6:%.*]] = atomicrmw min ptr [[PTR6:%.+]], i32 [[VAL6:.+]] syncscope("agent-one-as") monotonic, align 4
// CHECK:    [[TMP7:%.*]] = atomicrmw max ptr [[PTR7:%.+]], i32 [[VAL7:.+]] syncscope("agent-one-as") monotonic, align 4
void fi3b(int *a, int *b, int *c, int *d, int *e, int *f, int *g, int *h) {
  *a = __scoped_atomic_fetch_add(a, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
  *b = __scoped_atomic_fetch_sub(b, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
  *c = __scoped_atomic_fetch_and(c, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
  *d = __scoped_atomic_fetch_or(d, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
  *e = __scoped_atomic_fetch_xor(e, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
  *f = __scoped_atomic_fetch_nand(f, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
  *g = __scoped_atomic_fetch_min(g, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
  *h = __scoped_atomic_fetch_max(h, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
}

// CHECK-LABEL: define hidden void @fi3c(
// CHECK:    [[TMP0:%.*]] = atomicrmw add ptr [[PTR0:%.+]], i32 [[VAL0:.+]] syncscope("workgroup-one-as") monotonic, align 4
// CHECK:    [[TMP1:%.*]] = atomicrmw sub ptr [[PTR1:%.+]], i32 [[VAL1:.+]] syncscope("workgroup-one-as") monotonic, align 4
// CHECK:    [[TMP2:%.*]] = atomicrmw and ptr [[PTR2:%.+]], i32 [[VAL2:.+]] syncscope("workgroup-one-as") monotonic, align 4
// CHECK:    [[TMP3:%.*]] = atomicrmw or ptr [[PTR3:%.+]], i32 [[VAL3:.+]] syncscope("workgroup-one-as") monotonic, align 4
// CHECK:    [[TMP4:%.*]] = atomicrmw xor ptr [[PTR4:%.+]], i32 [[VAL4:.+]] syncscope("workgroup-one-as") monotonic, align 4
// CHECK:    [[TMP5:%.*]] = atomicrmw nand ptr [[PTR5:%.+]], i32 [[VAL5:.+]] syncscope("workgroup-one-as") monotonic, align 4
// CHECK:    [[TMP6:%.*]] = atomicrmw min ptr [[PTR6:%.+]], i32 [[VAL6:.+]] syncscope("workgroup-one-as") monotonic, align 4
// CHECK:    [[TMP7:%.*]] = atomicrmw max ptr [[PTR7:%.+]], i32 [[VAL7:.+]] syncscope("workgroup-one-as") monotonic, align 4
void fi3c(int *a, int *b, int *c, int *d, int *e, int *f, int *g, int *h) {
  *a = __scoped_atomic_fetch_add(a, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_WRKGRP);
  *b = __scoped_atomic_fetch_sub(b, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_WRKGRP);
  *c = __scoped_atomic_fetch_and(c, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_WRKGRP);
  *d = __scoped_atomic_fetch_or(d, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_WRKGRP);
  *e = __scoped_atomic_fetch_xor(e, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_WRKGRP);
  *f = __scoped_atomic_fetch_nand(f, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_WRKGRP);
  *g = __scoped_atomic_fetch_min(g, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_WRKGRP);
  *h = __scoped_atomic_fetch_max(h, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_WRKGRP);
}

// CHECK-LABEL: define hidden void @fi3d(
// CHECK:    [[TMP0:%.*]] = atomicrmw add ptr [[PTR0:%.+]], i32 [[VAL0:.+]] syncscope("wavefront-one-as") monotonic, align 4
// CHECK:    [[TMP1:%.*]] = atomicrmw sub ptr [[PTR1:%.+]], i32 [[VAL1:.+]] syncscope("wavefront-one-as") monotonic, align 4
// CHECK:    [[TMP2:%.*]] = atomicrmw and ptr [[PTR2:%.+]], i32 [[VAL2:.+]] syncscope("wavefront-one-as") monotonic, align 4
// CHECK:    [[TMP3:%.*]] = atomicrmw or ptr [[PTR3:%.+]], i32 [[VAL3:.+]] syncscope("wavefront-one-as") monotonic, align 4
// CHECK:    [[TMP4:%.*]] = atomicrmw xor ptr [[PTR4:%.+]], i32 [[VAL4:.+]] syncscope("wavefront-one-as") monotonic, align 4
// CHECK:    [[TMP5:%.*]] = atomicrmw nand ptr [[PTR5:%.+]], i32 [[VAL5:.+]] syncscope("wavefront-one-as") monotonic, align 4
// CHECK:    [[TMP6:%.*]] = atomicrmw min ptr [[PTR6:%.+]], i32 [[VAL6:.+]] syncscope("wavefront-one-as") monotonic, align 4
// CHECK:    [[TMP7:%.*]] = atomicrmw max ptr [[PTR7:%.+]], i32 [[VAL7:.+]] syncscope("wavefront-one-as") monotonic, align 4
void fi3d(int *a, int *b, int *c, int *d, int *e, int *f, int *g, int *h) {
  *a = __scoped_atomic_fetch_add(a, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_WVFRNT);
  *b = __scoped_atomic_fetch_sub(b, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_WVFRNT);
  *c = __scoped_atomic_fetch_and(c, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_WVFRNT);
  *d = __scoped_atomic_fetch_or(d, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_WVFRNT);
  *e = __scoped_atomic_fetch_xor(e, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_WVFRNT);
  *f = __scoped_atomic_fetch_nand(f, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_WVFRNT);
  *g = __scoped_atomic_fetch_min(g, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_WVFRNT);
  *h = __scoped_atomic_fetch_max(h, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_WVFRNT);
}

// CHECK-LABEL: define hidden void @fi3e(
// CHECK:    [[TMP0:%.*]] = atomicrmw add ptr [[PTR0:%.+]], i32 [[VAL0:.+]] syncscope("singlethread-one-as") monotonic, align 4
// CHECK:    [[TMP1:%.*]] = atomicrmw sub ptr [[PTR1:%.+]], i32 [[VAL1:.+]] syncscope("singlethread-one-as") monotonic, align 4
// CHECK:    [[TMP2:%.*]] = atomicrmw and ptr [[PTR2:%.+]], i32 [[VAL2:.+]] syncscope("singlethread-one-as") monotonic, align 4
// CHECK:    [[TMP3:%.*]] = atomicrmw or ptr [[PTR3:%.+]], i32 [[VAL3:.+]] syncscope("singlethread-one-as") monotonic, align 4
// CHECK:    [[TMP4:%.*]] = atomicrmw xor ptr [[PTR4:%.+]], i32 [[VAL4:.+]] syncscope("singlethread-one-as") monotonic, align 4
// CHECK:    [[TMP5:%.*]] = atomicrmw nand ptr [[PTR5:%.+]], i32 [[VAL5:.+]] syncscope("singlethread-one-as") monotonic, align 4
// CHECK:    [[TMP6:%.*]] = atomicrmw min ptr [[PTR6:%.+]], i32 [[VAL6:.+]] syncscope("singlethread-one-as") monotonic, align 4
// CHECK:    [[TMP7:%.*]] = atomicrmw max ptr [[PTR7:%.+]], i32 [[VAL7:.+]] syncscope("singlethread-one-as") monotonic, align 4
void fi3e(int *a, int *b, int *c, int *d, int *e, int *f, int *g, int *h) {
  *a = __scoped_atomic_fetch_add(a, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SINGLE);
  *b = __scoped_atomic_fetch_sub(b, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SINGLE);
  *c = __scoped_atomic_fetch_and(c, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SINGLE);
  *d = __scoped_atomic_fetch_or(d, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SINGLE);
  *e = __scoped_atomic_fetch_xor(e, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SINGLE);
  *f = __scoped_atomic_fetch_nand(f, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SINGLE);
  *g = __scoped_atomic_fetch_min(g, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SINGLE);
  *h = __scoped_atomic_fetch_max(h, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_SINGLE);
}

// CHECK-LABEL: define hidden zeroext i1 @fi4a(
// CHECK:    [[TMP0:%.*]] = cmpxchg ptr [[PTR0:%.+]], i32 [[VAL0:.+]], i32 [[VAL1:.+]] syncscope("one-as") acquire acquire, align 4
_Bool fi4a(int *i) {
  int cmp = 0;
  int desired = 1;
  return __scoped_atomic_compare_exchange(i, &cmp, &desired, 0,
                                          __ATOMIC_ACQUIRE, __ATOMIC_ACQUIRE,
                                          __MEMORY_SCOPE_SYSTEM);
}

// CHECK-LABEL: define hidden zeroext i1 @fi4b(
// CHECK:    [[TMP0:%.*]] = cmpxchg ptr [[PTR0:%.+]], i32 [[VAL0:.+]], i32 [[VAL1:.+]] syncscope("agent-one-as") acquire acquire, align 4
_Bool fi4b(int *i) {
  int cmp = 0;
  int desired = 1;
  return __scoped_atomic_compare_exchange(i, &cmp, &desired, 0,
                                          __ATOMIC_ACQUIRE, __ATOMIC_ACQUIRE,
                                          __MEMORY_SCOPE_DEVICE);
}

// CHECK-LABEL: define hidden zeroext i1 @fi4c(
// CHECK:    [[TMP0:%.*]] = cmpxchg ptr [[PTR0:%.+]], i32 [[VAL0:.+]], i32 [[VAL1:.+]] syncscope("workgroup-one-as") acquire acquire, align 4
_Bool fi4c(int *i) {
  int cmp = 0;
  int desired = 1;
  return __scoped_atomic_compare_exchange(i, &cmp, &desired, 0,
                                          __ATOMIC_ACQUIRE, __ATOMIC_ACQUIRE,
                                          __MEMORY_SCOPE_WRKGRP);
}

// CHECK-LABEL: define hidden zeroext i1 @fi4d(
// CHECK:    [[TMP0:%.*]] = cmpxchg ptr [[PTR0:%.+]], i32 [[VAL0:.+]], i32 [[VAL1:.+]] syncscope("wavefront-one-as") acquire acquire, align 4
_Bool fi4d(int *i) {
  int cmp = 0;
  int desired = 1;
  return __scoped_atomic_compare_exchange(i, &cmp, &desired, 0,
                                          __ATOMIC_ACQUIRE, __ATOMIC_ACQUIRE,
                                          __MEMORY_SCOPE_WVFRNT);
}

// CHECK-LABEL: define hidden zeroext i1 @fi4e(
// CHECK:    [[TMP0:%.*]] = cmpxchg ptr [[PTR0:%.+]], i32 [[VAL0:.+]], i32 [[VAL1:.+]] syncscope("singlethread-one-as") acquire acquire, align 4
_Bool fi4e(int *i) {
  int cmp = 0;
  int desired = 1;
  return __scoped_atomic_compare_exchange(i, &cmp, &desired, 0,
                                          __ATOMIC_ACQUIRE, __ATOMIC_ACQUIRE,
                                          __MEMORY_SCOPE_SINGLE);
}

// CHECK-LABEL: define hidden zeroext i1 @fi5a(
// CHECK:    [[TMP0:%.*]] = cmpxchg weak ptr [[PTR0:%.+]], i32 [[VAL0:.+]], i32 [[VAL1:.+]] syncscope("one-as") acquire acquire, align 4
_Bool fi5a(int *i) {
  int cmp = 0;
  return __scoped_atomic_compare_exchange_n(i, &cmp, 1, 1, __ATOMIC_ACQUIRE,
                                            __ATOMIC_ACQUIRE,
                                            __MEMORY_SCOPE_SYSTEM);
}

// CHECK-LABEL: define hidden zeroext i1 @fi5b(
// CHECK:    [[TMP0:%.*]] = cmpxchg weak ptr [[PTR0:%.+]], i32 [[VAL0:.+]], i32 [[VAL1:.+]] syncscope("agent-one-as") acquire acquire, align 4
_Bool fi5b(int *i) {
  int cmp = 0;
  return __scoped_atomic_compare_exchange_n(i, &cmp, 1, 1, __ATOMIC_ACQUIRE,
                                            __ATOMIC_ACQUIRE,
                                            __MEMORY_SCOPE_DEVICE);
}

// CHECK-LABEL: define hidden zeroext i1 @fi5c(
// CHECK:    [[TMP0:%.*]] = cmpxchg weak ptr [[PTR0:%.+]], i32 [[VAL0:.+]], i32 [[VAL1:.+]] syncscope("workgroup-one-as") acquire acquire, align 4
_Bool fi5c(int *i) {
  int cmp = 0;
  return __scoped_atomic_compare_exchange_n(
      i, &cmp, 1, 1, __ATOMIC_ACQUIRE, __ATOMIC_ACQUIRE, __MEMORY_SCOPE_WRKGRP);
}

// CHECK-LABEL: define hidden zeroext i1 @fi5d(
// CHECK:    [[TMP0:%.*]] = cmpxchg weak ptr [[PTR0:%.+]], i32 [[VAL0:.+]], i32 [[VAL1:.+]] syncscope("wavefront-one-as") acquire acquire, align 4
_Bool fi5d(int *i) {
  int cmp = 0;
  return __scoped_atomic_compare_exchange_n(
      i, &cmp, 1, 1, __ATOMIC_ACQUIRE, __ATOMIC_ACQUIRE, __MEMORY_SCOPE_WVFRNT);
}

// CHECK-LABEL: define hidden zeroext i1 @fi5e(
// CHECK:    [[TMP0:%.*]] = cmpxchg weak ptr [[PTR0:%.+]], i32 [[VAL0:.+]], i32 [[VAL1:.+]] syncscope("singlethread-one-as") acquire acquire, align 4
_Bool fi5e(int *i) {
  int cmp = 0;
  return __scoped_atomic_compare_exchange_n(
      i, &cmp, 1, 1, __ATOMIC_ACQUIRE, __ATOMIC_ACQUIRE, __MEMORY_SCOPE_SINGLE);
}

// CHECK-LABEL: define hidden i32 @fi6a(
// CHECK:    [[TMP0:%.*]] = atomicrmw xchg ptr [[PTR0:%.+]], i32 [[VAL0:.+]] syncscope("one-as") monotonic, align 4
int fi6a(int *c, int *d) {
  int ret;
  __scoped_atomic_exchange(c, d, &ret, __ATOMIC_RELAXED, __MEMORY_SCOPE_SYSTEM);
  return ret;
}

// CHECK-LABEL: define hidden i32 @fi6b(
// CHECK:    [[TMP0:%.*]] = atomicrmw xchg ptr [[PTR0:%.+]], i32 [[VAL0:.+]] syncscope("agent-one-as") monotonic, align 4
int fi6b(int *c, int *d) {
  int ret;
  __scoped_atomic_exchange(c, d, &ret, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
  return ret;
}

// CHECK-LABEL: define hidden i32 @fi6c(
// CHECK:    [[TMP0:%.*]] = atomicrmw xchg ptr [[PTR0:%.+]], i32 [[VAL0:.+]] syncscope("workgroup-one-as") monotonic, align 4
int fi6c(int *c, int *d) {
  int ret;
  __scoped_atomic_exchange(c, d, &ret, __ATOMIC_RELAXED, __MEMORY_SCOPE_WRKGRP);
  return ret;
}

// CHECK-LABEL: define hidden i32 @fi6d(
// CHECK:    [[TMP0:%.*]] = atomicrmw xchg ptr [[PTR0:%.+]], i32 [[VAL0:.+]] syncscope("wavefront-one-as") monotonic, align 4
int fi6d(int *c, int *d) {
  int ret;
  __scoped_atomic_exchange(c, d, &ret, __ATOMIC_RELAXED, __MEMORY_SCOPE_WVFRNT);
  return ret;
}

// CHECK-LABEL: define hidden i32 @fi6e(
// CHECK:    [[TMP0:%.*]] = atomicrmw xchg ptr [[PTR0:%.+]], i32 [[VAL0:.+]] syncscope("singlethread-one-as") monotonic, align 4
int fi6e(int *c, int *d) {
  int ret;
  __scoped_atomic_exchange(c, d, &ret, __ATOMIC_RELAXED, __MEMORY_SCOPE_SINGLE);
  return ret;
}

// CHECK-LABEL: define hidden zeroext i1 @fi7a(
// CHECK:    [[TMP0:%.*]] = atomicrmw xchg ptr [[PTR0:%.+]], i8 [[VAL0:.+]] syncscope("one-as") monotonic, align 1
_Bool fi7a(_Bool *c) {
  return __scoped_atomic_exchange_n(c, 1, __ATOMIC_RELAXED,
                                    __MEMORY_SCOPE_SYSTEM);
}

// CHECK-LABEL: define hidden zeroext i1 @fi7b(
// CHECK:    [[TMP0:%.*]] = atomicrmw xchg ptr [[PTR0:%.+]], i8 [[VAL0:.+]] syncscope("agent-one-as") monotonic, align 1
_Bool fi7b(_Bool *c) {
  return __scoped_atomic_exchange_n(c, 1, __ATOMIC_RELAXED,
                                    __MEMORY_SCOPE_DEVICE);
}

// CHECK-LABEL: define hidden zeroext i1 @fi7c(
// CHECK:    [[TMP0:%.*]] = atomicrmw xchg ptr [[PTR0:%.+]], i8 [[VAL0:.+]] syncscope("workgroup-one-as") monotonic, align 1
_Bool fi7c(_Bool *c) {
  return __scoped_atomic_exchange_n(c, 1, __ATOMIC_RELAXED,
                                    __MEMORY_SCOPE_WRKGRP);
}

// CHECK-LABEL: define hidden zeroext i1 @fi7d(
// CHECK:    [[TMP0:%.*]] = atomicrmw xchg ptr [[PTR0:%.+]], i8 [[VAL0:.+]] syncscope("wavefront-one-as") monotonic, align 1
_Bool fi7d(_Bool *c) {
  return __scoped_atomic_exchange_n(c, 1, __ATOMIC_RELAXED,
                                    __MEMORY_SCOPE_WVFRNT);
}

// CHECK-LABEL: define hidden zeroext i1 @fi7e(
// CHECK:    [[TMP0:%.*]] = atomicrmw xchg ptr [[PTR0:%.+]], i8 [[VAL0:.+]] syncscope("singlethread-one-as") monotonic, align 1
_Bool fi7e(_Bool *c) {
  return __scoped_atomic_exchange_n(c, 1, __ATOMIC_RELAXED, 
                                    __MEMORY_SCOPE_SINGLE);
}
