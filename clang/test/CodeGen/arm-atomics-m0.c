// RUN: %clang_cc1 %s -emit-llvm -o - -triple=thumbv6m-none--eabi -target-cpu cortex-m0 | FileCheck %s

int i;
long long l;

typedef enum memory_order {
  memory_order_relaxed, memory_order_consume, memory_order_acquire,
  memory_order_release, memory_order_acq_rel, memory_order_seq_cst
} memory_order;

void test_presence(void)
{
  // CHECK-LABEL: @test_presence
  // CHECK: atomicrmw add ptr {{.*}} seq_cst, align 4
  __atomic_fetch_add(&i, 1, memory_order_seq_cst);
  // CHECK: atomicrmw sub {{.*}} seq_cst, align 4
  __atomic_fetch_sub(&i, 1, memory_order_seq_cst);
  // CHECK: load atomic i32, ptr {{.*}} seq_cst, align 4
  int r;
  __atomic_load(&i, &r, memory_order_seq_cst);
  // CHECK: store atomic i32 {{.*}}, ptr {{.*}} seq_cst, align 4
  r = 0;
  __atomic_store(&i, &r, memory_order_seq_cst);

  // CHECK: atomicrmw add {{.*}} seq_cst, align 8
  __atomic_fetch_add(&l, 1, memory_order_seq_cst);
  // CHECK: atomicrmw sub {{.*}} seq_cst, align 8
  __atomic_fetch_sub(&l, 1, memory_order_seq_cst);
  // CHECK: load atomic i64, ptr {{.*}} seq_cst, align 8
  long long rl;
  __atomic_load(&l, &rl, memory_order_seq_cst);
  // CHECK: store atomic i64 {{.*}}, ptr {{.*}} seq_cst, align 8
  rl = 0;
  __atomic_store(&l, &rl, memory_order_seq_cst);
}
