// RUN: %clang_cc1 %s -cl-std=CL2.0 -emit-llvm -O0 -o - -triple=amdgcn-amd-amdhsa \
// RUN:   | FileCheck %s

// Also test serialization of atomic operations here, to avoid duplicating the test.
// RUN: %clang_cc1 %s -cl-std=CL2.0 -emit-pch -O0 -o %t -triple=amdgcn-amd-amdhsa
// RUN: %clang_cc1 %s -cl-std=CL2.0 -include-pch %t -O0 -triple=amdgcn-amd-amdhsa \
// RUN:   -emit-llvm -o - | FileCheck %s

#ifndef ALREADY_INCLUDED
#define ALREADY_INCLUDED

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

typedef __INTPTR_TYPE__ intptr_t;
typedef int int8 __attribute__((ext_vector_type(8)));

typedef enum memory_order {
  memory_order_relaxed = __ATOMIC_RELAXED,
  memory_order_acquire = __ATOMIC_ACQUIRE,
  memory_order_release = __ATOMIC_RELEASE,
  memory_order_acq_rel = __ATOMIC_ACQ_REL,
  memory_order_seq_cst = __ATOMIC_SEQ_CST
} memory_order;

typedef enum memory_scope {
  memory_scope_work_item = __OPENCL_MEMORY_SCOPE_WORK_ITEM,
  memory_scope_work_group = __OPENCL_MEMORY_SCOPE_WORK_GROUP,
  memory_scope_device = __OPENCL_MEMORY_SCOPE_DEVICE,
  memory_scope_all_svm_devices = __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES,
#if defined(cl_intel_subgroups) || defined(cl_khr_subgroups)
  memory_scope_sub_group = __OPENCL_MEMORY_SCOPE_SUB_GROUP
#endif
} memory_scope;

atomic_int j;

void fi1(atomic_int *i) {
  // CHECK-LABEL: @fi1
  // CHECK: load atomic i32, ptr %{{[.0-9A-Z_a-z]+}} syncscope("workgroup") seq_cst, align 4{{$}}
  int x = __opencl_atomic_load(i, memory_order_seq_cst, memory_scope_work_group);

  // CHECK: load atomic i32, ptr %{{[.0-9A-Z_a-z]+}} syncscope("agent") seq_cst, align 4{{$}}
  x = __opencl_atomic_load(i, memory_order_seq_cst, memory_scope_device);

  // CHECK: load atomic i32, ptr %{{[.0-9A-Z_a-z]+}} seq_cst, align 4{{$}}
  x = __opencl_atomic_load(i, memory_order_seq_cst, memory_scope_all_svm_devices);

  // CHECK: load atomic i32, ptr %{{[.0-9A-Z_a-z]+}} syncscope("wavefront") seq_cst, align 4{{$}}
  x = __opencl_atomic_load(i, memory_order_seq_cst, memory_scope_sub_group);
}

void fi2(atomic_int *i) {
  // CHECK-LABEL: @fi2
  // CHECK: store atomic i32 %{{[.0-9A-Z_a-z]+}}, ptr %{{[.0-9A-Z_a-z]+}} syncscope("workgroup") seq_cst, align 4{{$}}
  __opencl_atomic_store(i, 1, memory_order_seq_cst, memory_scope_work_group);
}

void test_addr(global atomic_int *ig, private atomic_int *ip, local atomic_int *il) {
  // CHECK-LABEL: @test_addr
  // CHECK: store atomic i32 %{{[.0-9A-Z_a-z]+}}, ptr addrspace(1) %{{[.0-9A-Z_a-z]+}} syncscope("workgroup") seq_cst, align 4{{$}}
  __opencl_atomic_store(ig, 1, memory_order_seq_cst, memory_scope_work_group);

  // CHECK: store atomic i32 %{{[.0-9A-Z_a-z]+}}, ptr addrspace(5) %{{[.0-9A-Z_a-z]+}} syncscope("workgroup") seq_cst, align 4{{$}}
  __opencl_atomic_store(ip, 1, memory_order_seq_cst, memory_scope_work_group);

  // CHECK: store atomic i32 %{{[.0-9A-Z_a-z]+}}, ptr addrspace(3) %{{[.0-9A-Z_a-z]+}} syncscope("workgroup") seq_cst, align 4{{$}}
  __opencl_atomic_store(il, 1, memory_order_seq_cst, memory_scope_work_group);
}

void fi3(atomic_int *i, atomic_uint *ui) {
  // CHECK-LABEL: @fi3
  // CHECK: atomicrmw and ptr %{{[.0-9A-Z_a-z]+}}, i32 %{{[.0-9A-Z_a-z]+}} syncscope("workgroup") seq_cst, align 4, !noalias.addrspace [[$NOPRIVATE:![0-9]+]], [[$DEFMD:!amdgpu.no.fine.grained.memory ![0-9]+, !amdgpu.no.remote.memory ![0-9]+$]]
  int x = __opencl_atomic_fetch_and(i, 1, memory_order_seq_cst, memory_scope_work_group);

  // CHECK: atomicrmw min ptr %{{[.0-9A-Z_a-z]+}}, i32 %{{[.0-9A-Z_a-z]+}} syncscope("workgroup") seq_cst, align 4, !noalias.addrspace [[$NOPRIVATE]], [[$DEFMD]]
  x = __opencl_atomic_fetch_min(i, 1, memory_order_seq_cst, memory_scope_work_group);

  // CHECK: atomicrmw max ptr %{{[.0-9A-Z_a-z]+}}, i32 %{{[.0-9A-Z_a-z]+}} syncscope("workgroup") seq_cst, align 4, !noalias.addrspace [[$NOPRIVATE]], [[$DEFMD]]
  x = __opencl_atomic_fetch_max(i, 1, memory_order_seq_cst, memory_scope_work_group);

  // CHECK: atomicrmw umin ptr %{{[.0-9A-Z_a-z]+}}, i32 %{{[.0-9A-Z_a-z]+}} syncscope("workgroup") seq_cst, align 4, !noalias.addrspace [[$NOPRIVATE]], [[$DEFMD]]
  x = __opencl_atomic_fetch_min(ui, 1, memory_order_seq_cst, memory_scope_work_group);

  // CHECK: atomicrmw umax ptr %{{[.0-9A-Z_a-z]+}}, i32 %{{[.0-9A-Z_a-z]+}} syncscope("workgroup") seq_cst, align 4, !noalias.addrspace [[$NOPRIVATE]], [[$DEFMD]]
  x = __opencl_atomic_fetch_max(ui, 1, memory_order_seq_cst, memory_scope_work_group);
}

bool fi4(atomic_int *i) {
  // CHECK-LABEL: @fi4(
  // CHECK: [[PAIR:%[.0-9A-Z_a-z]+]] = cmpxchg ptr [[PTR:%[.0-9A-Z_a-z]+]], i32 [[EXPECTED:%[.0-9A-Z_a-z]+]], i32 [[DESIRED:%[.0-9A-Z_a-z]+]] syncscope("workgroup-one-as") acquire acquire, align 4, !noalias.addrspace [[$NOPRIVATE]]{{$}}
  // CHECK: [[OLD:%[.0-9A-Z_a-z]+]] = extractvalue { i32, i1 } [[PAIR]], 0
  // CHECK: [[CMP:%[.0-9A-Z_a-z]+]] = extractvalue { i32, i1 } [[PAIR]], 1
  // CHECK: br i1 [[CMP]], label %[[STORE_EXPECTED:[.0-9A-Z_a-z]+]], label %[[CONTINUE:[.0-9A-Z_a-z]+]]
  // CHECK: store i32 [[OLD]]
  int cmp = 0;
  return __opencl_atomic_compare_exchange_strong(i, &cmp, 1, memory_order_acquire, memory_order_acquire, memory_scope_work_group);
}

void fi5(atomic_int *i, int scope) {
  // CHECK-LABEL: @fi5
  // CHECK: switch i32 %{{.*}}, label %[[opencl_allsvmdevices:.*]] [
  // CHECK-NEXT: i32 1, label %[[opencl_workgroup:.*]]
  // CHECK-NEXT: i32 2, label %[[opencl_device:.*]]
  // CHECK-NEXT: i32 4, label %[[opencl_subgroup:.*]]
  // CHECK-NEXT: ]
  // CHECK: [[opencl_workgroup]]:
  // CHECK: load atomic i32, ptr %{{.*}} syncscope("workgroup") seq_cst, align 4{{$}}
  // CHECK: br label %[[continue:.*]]
  // CHECK: [[opencl_device]]:
  // CHECK: load atomic i32, ptr %{{.*}} syncscope("agent") seq_cst, align 4{{$}}
  // CHECK: br label %[[continue]]
  // CHECK: [[opencl_allsvmdevices]]:
  // CHECK: load atomic i32, ptr %{{.*}} seq_cst, align 4
  // CHECK: br label %[[continue]]
  // CHECK: [[opencl_subgroup]]:
  // CHECK: load atomic i32, ptr %{{.*}} syncscope("wavefront") seq_cst, align 4{{$}}
  // CHECK: br label %[[continue]]
  // CHECK: [[continue]]:
  int x = __opencl_atomic_load(i, memory_order_seq_cst, scope);
}

void fi6(atomic_int *i, int order, int scope) {
  // CHECK-LABEL: @fi6
  // CHECK: switch i32 %{{.*}}, label %[[monotonic:.*]] [
  // CHECK-NEXT: i32 1, label %[[acquire:.*]]
  // CHECK-NEXT: i32 2, label %[[acquire:.*]]
  // CHECK-NEXT: i32 5, label %[[seqcst:.*]]
  // CHECK-NEXT: ]
  // CHECK: [[monotonic]]:
  // CHECK: switch i32 %{{.*}}, label %[[MON_ALL:.*]] [
  // CHECK-NEXT: i32 1, label %[[MON_WG:.*]]
  // CHECK-NEXT: i32 2, label %[[MON_DEV:.*]]
  // CHECK-NEXT: i32 4, label %[[MON_SUB:.*]]
  // CHECK-NEXT: ]
  // CHECK: [[acquire]]:
  // CHECK: switch i32 %{{.*}}, label %[[ACQ_ALL:.*]] [
  // CHECK-NEXT: i32 1, label %[[ACQ_WG:.*]]
  // CHECK-NEXT: i32 2, label %[[ACQ_DEV:.*]]
  // CHECK-NEXT: i32 4, label %[[ACQ_SUB:.*]]
  // CHECK-NEXT: ]
  // CHECK: [[seqcst]]:
  // CHECK: switch i32 %{{.*}}, label %[[SEQ_ALL:.*]] [
  // CHECK-NEXT: i32 1, label %[[SEQ_WG:.*]]
  // CHECK-NEXT: i32 2, label %[[SEQ_DEV:.*]]
  // CHECK-NEXT: i32 4, label %[[SEQ_SUB:.*]]
  // CHECK-NEXT: ]
  // CHECK: [[MON_WG]]:
  // CHECK: load atomic i32, ptr %{{.*}} syncscope("workgroup-one-as") monotonic, align 4{{$}}
  // CHECK: [[MON_DEV]]:
  // CHECK: load atomic i32, ptr %{{.*}} syncscope("agent-one-as") monotonic, align 4{{$}}
  // CHECK: [[MON_ALL]]:
  // CHECK: load atomic i32, ptr %{{.*}} monotonic, align 4{{$}}
  // CHECK: [[MON_SUB]]:
  // CHECK: load atomic i32, ptr %{{.*}} syncscope("wavefront-one-as") monotonic, align 4{{$}}
  // CHECK: [[ACQ_WG]]:
  // CHECK: load atomic i32, ptr %{{.*}} syncscope("workgroup-one-as") acquire, align 4{{$}}
  // CHECK: [[ACQ_DEV]]:
  // CHECK: load atomic i32, ptr %{{.*}} syncscope("agent-one-as") acquire, align 4{{$}}
  // CHECK: [[ACQ_ALL]]:
  // CHECK: load atomic i32, ptr %{{.*}} acquire, align 4{{$}}
  // CHECK: [[ACQ_SUB]]:
  // CHECK: load atomic i32, ptr %{{.*}} syncscope("wavefront-one-as") acquire, align 4{{$}}
  // CHECK: [[SEQ_WG]]:
  // CHECK: load atomic i32, ptr %{{.*}} syncscope("workgroup") seq_cst, align 4{{$}}
  // CHECK: [[SEQ_DEV]]:
  // CHECK: load atomic i32, ptr %{{.*}} syncscope("agent") seq_cst, align 4{{$}}
  // CHECK: [[SEQ_ALL]]:
  // CHECK: load atomic i32, ptr %{{.*}} seq_cst, align 4{{$}}
  // CHECK: [[SEQ_SUB]]:
  // CHECK: load atomic i32, ptr %{{.*}} syncscope("wavefront") seq_cst, align 4{{$}}
  int x = __opencl_atomic_load(i, order, scope);
}

float ff1(global atomic_float *d) {
  // CHECK-LABEL: @ff1
  // CHECK: load atomic i32, ptr addrspace(1) {{.*}} syncscope("workgroup-one-as") monotonic, align 4{{$}}
  return __opencl_atomic_load(d, memory_order_relaxed, memory_scope_work_group);
}

void ff2(atomic_float *d) {
  // CHECK-LABEL: @ff2
  // CHECK: store atomic i32 {{.*}} syncscope("workgroup-one-as") release, align 4
  __opencl_atomic_store(d, 1, memory_order_release, memory_scope_work_group);
}

float ff3(atomic_float *d) {
  // CHECK-LABEL: @ff3
  // CHECK: atomicrmw xchg ptr {{.*}} syncscope("workgroup") seq_cst, align 4, !noalias.addrspace [[$NOPRIVATE]], [[$DEFMD]]
  return __opencl_atomic_exchange(d, 2, memory_order_seq_cst, memory_scope_work_group);
}

float ff4(global atomic_float *d, float a) {
  // CHECK-LABEL: @ff4
  // CHECK: atomicrmw fadd ptr addrspace(1) {{.*}} syncscope("workgroup-one-as") monotonic, align 4, [[$DEFMD]]
  return __opencl_atomic_fetch_add(d, a, memory_order_relaxed, memory_scope_work_group);
}

float ff5(global atomic_double *d, double a) {
  // CHECK-LABEL: @ff5
  // CHECK: atomicrmw fadd ptr addrspace(1) {{.*}} syncscope("workgroup-one-as") monotonic, align 8, [[$DEFMD]]
  return __opencl_atomic_fetch_add(d, a, memory_order_relaxed, memory_scope_work_group);
}

float ff4_generic(atomic_float *d, float a) {
  // CHECK-LABEL: @ff4_generic
  // CHECK: atomicrmw fadd ptr {{.*}} syncscope("workgroup-one-as") monotonic, align 4, !noalias.addrspace [[$NOPRIVATE]], [[$DEFMD]]
  return __opencl_atomic_fetch_add(d, a, memory_order_relaxed, memory_scope_work_group);
}

float ff5_generic(atomic_double *d, double a) {
  // CHECK-LABEL: @ff5_generic
  // CHECK: atomicrmw fadd ptr {{.*}} syncscope("workgroup-one-as") monotonic, align 8, !noalias.addrspace [[$NOPRIVATE]], [[$DEFMD]]
  return __opencl_atomic_fetch_add(d, a, memory_order_relaxed, memory_scope_work_group);
}

// CHECK-LABEL: @atomic_init_foo
void atomic_init_foo()
{
  // CHECK-NOT: atomic
  // CHECK: store
  __opencl_atomic_init(&j, 42);

  // CHECK-NOT: atomic
  // CHECK: }
}

// CHECK-LABEL: @failureOrder
void failureOrder(atomic_int *ptr, int *ptr2) {
  // CHECK: cmpxchg ptr {{%[0-9A-Za-z._]+}}, i32 {{%[0-9A-Za-z._]+}}, i32 {{%[0-9A-Za-z_.]+}} syncscope("workgroup-one-as") acquire monotonic, align 4, !noalias.addrspace [[$NOPRIVATE]]{{$}}
  __opencl_atomic_compare_exchange_strong(ptr, ptr2, 43, memory_order_acquire, memory_order_relaxed, memory_scope_work_group);

  // CHECK: cmpxchg weak ptr {{%[0-9A-Za-z._]+}}, i32 {{%[0-9A-Za-z._]+}}, i32 {{%[0-9A-Za-z_.]+}} syncscope("workgroup") seq_cst acquire, align 4, !noalias.addrspace [[$NOPRIVATE]]{{$}}
  __opencl_atomic_compare_exchange_weak(ptr, ptr2, 43, memory_order_seq_cst, memory_order_acquire, memory_scope_work_group);
}

// CHECK-LABEL: @generalFailureOrder
void generalFailureOrder(atomic_int *ptr, int *ptr2, int success, int fail) {
  __opencl_atomic_compare_exchange_strong(ptr, ptr2, 42, success, fail, memory_scope_work_group);
// CHECK: switch i32 {{.*}}, label %[[MONOTONIC:[0-9a-zA-Z._]+]] [
  // CHECK-NEXT: i32 1, label %[[ACQUIRE:[0-9a-zA-Z._]+]]
  // CHECK-NEXT: i32 2, label %[[ACQUIRE]]
  // CHECK-NEXT: i32 3, label %[[RELEASE:[0-9a-zA-Z._]+]]
  // CHECK-NEXT: i32 4, label %[[ACQREL:[0-9a-zA-Z._]+]]
  // CHECK-NEXT: i32 5, label %[[SEQCST:[0-9a-zA-Z._]+]]

  // CHECK: [[MONOTONIC]]
  // CHECK: switch {{.*}}, label %[[MONOTONIC_MONOTONIC:[0-9a-zA-Z._]+]] [
  // CHECK-NEXT: i32 1, label %[[MONOTONIC_ACQUIRE:[0-9a-zA-Z._]+]]
  // CHECK-NEXT: i32 2, label %[[MONOTONIC_ACQUIRE:[0-9a-zA-Z._]+]]
  // CHECK-NEXT: i32 5, label %[[MONOTONIC_SEQCST:[0-9a-zA-Z._]+]]
  // CHECK-NEXT: ]

  // CHECK: [[ACQUIRE]]
  // CHECK: switch {{.*}}, label %[[ACQUIRE_MONOTONIC:[0-9a-zA-Z._]+]] [
  // CHECK-NEXT: i32 1, label %[[ACQUIRE_ACQUIRE:[0-9a-zA-Z._]+]]
  // CHECK-NEXT: i32 2, label %[[ACQUIRE_ACQUIRE:[0-9a-zA-Z._]+]]
  // CHECK-NEXT: i32 5, label %[[ACQUIRE_SEQCST:[0-9a-zA-Z._]+]]
  // CHECK-NEXT: ]

  // CHECK: [[RELEASE]]
  // CHECK: switch {{.*}}, label %[[RELEASE_MONOTONIC:[0-9a-zA-Z._]+]] [
  // CHECK-NEXT: i32 1, label %[[RELEASE_ACQUIRE:[0-9a-zA-Z._]+]]
  // CHECK-NEXT: i32 2, label %[[RELEASE_ACQUIRE:[0-9a-zA-Z._]+]]
  // CHECK-NEXT: i32 5, label %[[RELEASE_SEQCST:[0-9a-zA-Z._]+]]
  // CHECK-NEXT: ]

  // CHECK: [[ACQREL]]
  // CHECK: switch {{.*}}, label %[[ACQREL_MONOTONIC:[0-9a-zA-Z._]+]] [
  // CHECK-NEXT: i32 1, label %[[ACQREL_ACQUIRE:[0-9a-zA-Z._]+]]
  // CHECK-NEXT: i32 2, label %[[ACQREL_ACQUIRE:[0-9a-zA-Z._]+]]
  // CHECK-NEXT: i32 5, label %[[ACQREL_SEQCST:[0-9a-zA-Z._]+]]
  // CHECK-NEXT: ]

  // CHECK: [[SEQCST]]
  // CHECK: switch {{.*}}, label %[[SEQCST_MONOTONIC:[0-9a-zA-Z._]+]] [
  // CHECK-NEXT: i32 1, label %[[SEQCST_ACQUIRE:[0-9a-zA-Z._]+]]
  // CHECK-NEXT: i32 2, label %[[SEQCST_ACQUIRE]]
  // CHECK-NEXT: i32 5, label %[[SEQCST_SEQCST:[0-9a-zA-Z._]+]]
  // CHECK-NEXT: ]

  // CHECK: [[MONOTONIC_MONOTONIC]]
  // CHECK: cmpxchg {{.*}} monotonic monotonic, align 4, !noalias.addrspace [[$NOPRIVATE]]{{$}}
  // CHECK: br

  // CHECK: [[MONOTONIC_ACQUIRE]]
  // CHECK: cmpxchg {{.*}} monotonic acquire, align 4, !noalias.addrspace [[$NOPRIVATE]]{{$}}
  // CHECK: br

  // CHECK: [[MONOTONIC_SEQCST]]
  // CHECK: cmpxchg {{.*}} monotonic seq_cst, align 4, !noalias.addrspace [[$NOPRIVATE]]{{$}}
  // CHECK: br

  // CHECK: [[ACQUIRE_MONOTONIC]]
  // CHECK: cmpxchg {{.*}} acquire monotonic, align 4, !noalias.addrspace [[$NOPRIVATE]]{{$}}
  // CHECK: br

  // CHECK: [[ACQUIRE_ACQUIRE]]
  // CHECK: cmpxchg {{.*}} acquire acquire, align 4, !noalias.addrspace [[$NOPRIVATE]]{{$}}
  // CHECK: br

  // CHECK: [[ACQUIRE_SEQCST]]
  // CHECK: cmpxchg {{.*}} acquire seq_cst, align 4, !noalias.addrspace [[$NOPRIVATE]]{{$}}
  // CHECK: br

  // CHECK: [[RELEASE_MONOTONIC]]
  // CHECK: cmpxchg {{.*}} release monotonic, align 4, !noalias.addrspace [[$NOPRIVATE]]{{$}}
  // CHECK: br

  // CHECK: [[RELEASE_ACQUIRE]]
  // CHECK: cmpxchg {{.*}} release acquire, align 4, !noalias.addrspace [[$NOPRIVATE]]{{$}}
  // CHECK: br

  // CHECK: [[RELEASE_SEQCST]]
  // CHECK: cmpxchg {{.*}} release seq_cst, align 4, !noalias.addrspace [[$NOPRIVATE]]{{$}}
  // CHECK: br

  // CHECK: [[ACQREL_MONOTONIC]]
  // CHECK: cmpxchg {{.*}} acq_rel monotonic, align 4, !noalias.addrspace [[$NOPRIVATE]]{{$}}
  // CHECK: br

  // CHECK: [[ACQREL_ACQUIRE]]
  // CHECK: cmpxchg {{.*}} acq_rel acquire, align 4, !noalias.addrspace [[$NOPRIVATE]]{{$}}
  // CHECK: br

  // CHECK: [[ACQREL_SEQCST]]
  // CHECK: cmpxchg {{.*}} acq_rel seq_cst, align 4, !noalias.addrspace [[$NOPRIVATE]]{{$}}
  // CHECK: br

  // CHECK: [[SEQCST_MONOTONIC]]
  // CHECK: cmpxchg {{.*}} seq_cst monotonic, align 4, !noalias.addrspace [[$NOPRIVATE]]{{$}}
  // CHECK: br

  // CHECK: [[SEQCST_ACQUIRE]]
  // CHECK: cmpxchg {{.*}} seq_cst acquire, align 4, !noalias.addrspace [[$NOPRIVATE]]{{$}}
  // CHECK: br

  // CHECK: [[SEQCST_SEQCST]]
  // CHECK: cmpxchg {{.*}} seq_cst seq_cst, align 4, !noalias.addrspace [[$NOPRIVATE]]{{$}}
  // CHECK: br
}

int test_volatile(volatile atomic_int *i) {
  // CHECK-LABEL: @test_volatile
  // CHECK:      %[[i_addr:.*]] = alloca ptr
  // CHECK-NEXT: %[[atomicdst:.*]] = alloca i32
  // CHECK-NEXT: %[[retval_ascast:.*]] = addrspacecast ptr addrspace(5) {{.*}} to ptr
  // CHECK-NEXT: %[[i_addr_ascast:.*]] = addrspacecast ptr addrspace(5) %[[i_addr]] to ptr
  // CHECK-NEXT: %[[atomicdst_ascast:.*]] = addrspacecast ptr addrspace(5) %[[atomicdst]] to ptr
  // CHECK-NEXT: store ptr %i, ptr %[[i_addr_ascast]]
  // CHECK-NEXT: %[[addr:.*]] = load ptr, ptr %[[i_addr_ascast]]
  // CHECK-NEXT: %[[res:.*]] = load atomic volatile i32, ptr %[[addr]] syncscope("workgroup") seq_cst, align 4{{$}}
  // CHECK-NEXT: store i32 %[[res]], ptr %[[atomicdst_ascast]]
  // CHECK-NEXT: %[[retval:.*]] = load i32, ptr %[[atomicdst_ascast]]
  // CHECK-NEXT: ret i32 %[[retval]]
  return __opencl_atomic_load(i, memory_order_seq_cst, memory_scope_work_group);
}

#endif

// CHECK: [[$NOPRIVATE]] = !{i32 5, i32 6}
