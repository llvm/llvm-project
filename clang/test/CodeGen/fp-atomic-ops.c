// RUN: %clang_cc1 %s -emit-llvm -DDOUBLE -O0 -o - -triple=amdgcn-amd-amdhsa \
// RUN:   | FileCheck -check-prefixes=FLOAT,DOUBLE %s

// RUN: %clang_cc1 %s -emit-llvm -DDOUBLE -O0 -o - -triple=aarch64-linux-gnu \
// RUN:   | FileCheck -check-prefixes=FLOAT,DOUBLE %s

// RUN: %clang_cc1 %s -emit-llvm -O0 -o - -triple=armv8-apple-ios7.0 \
// RUN:   | FileCheck -check-prefixes=FLOAT %s

// RUN: %clang_cc1 %s -emit-llvm -DDOUBLE -O0 -o - -triple=hexagon \
// RUN:   | FileCheck -check-prefixes=FLOAT,DOUBLE %s

// RUN: %clang_cc1 %s -emit-llvm -DDOUBLE -O0 -o - -triple=mips64-mti-linux-gnu \
// RUN:   | FileCheck -check-prefixes=FLOAT,DOUBLE %s

// RUN: %clang_cc1 %s -emit-llvm -O0 -o - -triple=i686-linux-gnu \
// RUN:   | FileCheck -check-prefixes=FLOAT %s

// RUN: %clang_cc1 %s -emit-llvm -DDOUBLE -O0 -o - -triple=x86_64-linux-gnu \
// RUN:   | FileCheck -check-prefixes=FLOAT,DOUBLE %s

typedef enum memory_order {
  memory_order_relaxed = __ATOMIC_RELAXED,
  memory_order_acquire = __ATOMIC_ACQUIRE,
  memory_order_release = __ATOMIC_RELEASE,
  memory_order_acq_rel = __ATOMIC_ACQ_REL,
  memory_order_seq_cst = __ATOMIC_SEQ_CST
} memory_order;

typedef float float2 __attribute__((ext_vector_type(2)));
typedef double double2 __attribute__((ext_vector_type(2)));

void test(float *f, float ff, double *d, double dd) {
  // FLOAT: atomicrmw fadd ptr {{.*}} monotonic
  __atomic_fetch_add(f, ff, memory_order_relaxed);

  // FLOAT: atomicrmw fsub ptr {{.*}} monotonic
  __atomic_fetch_sub(f, ff, memory_order_relaxed);

#ifdef DOUBLE
  // DOUBLE: atomicrmw fadd ptr {{.*}} monotonic
  __atomic_fetch_add(d, dd, memory_order_relaxed);

  // DOUBLE: atomicrmw fsub ptr {{.*}} monotonic
  __atomic_fetch_sub(d, dd, memory_order_relaxed);
#endif
}

typedef float float2 __attribute__((ext_vector_type(2)));
typedef double double2 __attribute__((ext_vector_type(2)));

void test_vector(float2 *f, float2 ff, double2 *d, double2 dd) {
  // FLOAT: atomicrmw fadd ptr {{.*}} <2 x float> {{.*}} monotonic
  __atomic_fetch_add(f, ff, memory_order_relaxed);

  // FLOAT: atomicrmw fsub ptr {{.*}} <2 x float> {{.*}} monotonic
  __atomic_fetch_sub(f, ff, memory_order_relaxed);

#ifdef DOUBLE
  // DOUBLE: atomicrmw fadd ptr {{.*}} <2 x double> {{.*}}  monotonic
  __atomic_fetch_add(d, dd, memory_order_relaxed);

  // DOUBLE: atomicrmw fsub ptr {{.*}} <2 x double> {{.*}} monotonic
  __atomic_fetch_sub(d, dd, memory_order_relaxed);
#endif
}
