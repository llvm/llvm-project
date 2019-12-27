// RUN: %clang_cc1 %s -cl-std=CL2.0 -verify -fsyntax-only -triple=spir64
// RUN: %clang_cc1 %s -cl-std=CL2.0 -verify -fsyntax-only -triple=amdgcn-amdhsa-amd-opencl

// Basic parsing/Sema tests for __opencl_atomic_*

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

struct S { char c[3]; };

char i8;
short i16;
int i32;
int8 i64;

atomic_int gn;
void f(atomic_int *i, const atomic_int *ci,
       atomic_intptr_t *p, atomic_float *d,
       int *I, const int *CI,
       intptr_t *P, float *D, struct S *s1, struct S *s2,
       global atomic_int *i_g, local atomic_int *i_l, private atomic_int *i_p,
       constant atomic_int *i_c) {
  __opencl_atomic_init(I, 5); // expected-error {{address argument to atomic operation must be a pointer to _Atomic type ('__generic int *' invalid)}}
  __opencl_atomic_init(ci, 5); // expected-error {{address argument to atomic operation must be a pointer to non-const _Atomic type ('const __generic atomic_int *' (aka 'const __generic _Atomic(int) *') invalid)}}

  __opencl_atomic_load(0); // expected-error {{too few arguments to function call, expected 3, have 1}}
  __opencl_atomic_load(0, 0, 0, 0); // expected-error {{too many arguments to function call, expected 3, have 4}}
  __opencl_atomic_store(0,0,0,0); // expected-error {{address argument to atomic builtin must be a pointer}}
  __opencl_atomic_store((int *)0, 0, 0, 0); // expected-error {{address argument to atomic operation must be a pointer to _Atomic type ('__generic int *' invalid)}}
  __opencl_atomic_store(i, 0, memory_order_relaxed, memory_scope_work_group);
  __opencl_atomic_store(ci, 0, memory_order_relaxed, memory_scope_work_group); // expected-error {{address argument to atomic operation must be a pointer to non-const _Atomic type ('const __generic atomic_int *' (aka 'const __generic _Atomic(int) *') invalid)}}
  __opencl_atomic_store(i_g, 0, memory_order_relaxed, memory_scope_work_group);
  __opencl_atomic_store(i_l, 0, memory_order_relaxed, memory_scope_work_group);
  __opencl_atomic_store(i_p, 0, memory_order_relaxed, memory_scope_work_group);
  __opencl_atomic_store(i_c, 0, memory_order_relaxed, memory_scope_work_group); // expected-error {{address argument to atomic operation must be a pointer to non-constant _Atomic type ('__constant atomic_int *' (aka '__constant _Atomic(int) *') invalid)}}

  __opencl_atomic_load(i, memory_order_seq_cst, memory_scope_work_group);
  __opencl_atomic_load(p, memory_order_seq_cst, memory_scope_work_group);
  __opencl_atomic_load(d, memory_order_seq_cst, memory_scope_work_group);
  __opencl_atomic_load(ci, memory_order_seq_cst, memory_scope_work_group);
  __opencl_atomic_load(i_c, memory_order_seq_cst, memory_scope_work_group); // expected-error {{address argument to atomic operation must be a pointer to non-constant _Atomic type ('__constant atomic_int *' (aka '__constant _Atomic(int) *') invalid)}}

  __opencl_atomic_store(i, 1, memory_order_seq_cst, memory_scope_work_group);
  __opencl_atomic_store(p, 1, memory_order_seq_cst, memory_scope_work_group);
  (int)__opencl_atomic_store(d, 1, memory_order_seq_cst, memory_scope_work_group); // expected-error {{operand of type 'void' where arithmetic or pointer type is required}}

  int exchange_1 = __opencl_atomic_exchange(i, 1, memory_order_seq_cst, memory_scope_work_group);
  int exchange_2 = __opencl_atomic_exchange(I, 1, memory_order_seq_cst, memory_scope_work_group); // expected-error {{address argument to atomic operation must be a pointer to _Atomic}}

  __opencl_atomic_fetch_add(i, 1, memory_order_seq_cst, memory_scope_work_group);
  __opencl_atomic_fetch_add(p, 1, memory_order_seq_cst, memory_scope_work_group);
  __opencl_atomic_fetch_add(d, 1, memory_order_seq_cst, memory_scope_work_group); // expected-error {{address argument to atomic operation must be a pointer to atomic integer or pointer ('__generic atomic_float *' (aka '__generic _Atomic(float) *') invalid)}}
  __opencl_atomic_fetch_and(i, 1, memory_order_seq_cst, memory_scope_work_group);
  __opencl_atomic_fetch_and(p, 1, memory_order_seq_cst, memory_scope_work_group);
  __opencl_atomic_fetch_and(d, 1, memory_order_seq_cst, memory_scope_work_group); // expected-error {{address argument to atomic operation must be a pointer to atomic integer ('__generic atomic_float *' (aka '__generic _Atomic(float) *') invalid)}}

  __opencl_atomic_fetch_min(i, 1, memory_order_seq_cst, memory_scope_work_group);
  __opencl_atomic_fetch_max(i, 1, memory_order_seq_cst, memory_scope_work_group);
  __opencl_atomic_fetch_min(d, 1, memory_order_seq_cst, memory_scope_work_group); // expected-error {{address argument to atomic operation must be a pointer to atomic integer ('__generic atomic_float *' (aka '__generic _Atomic(float) *') invalid)}}
  __opencl_atomic_fetch_max(d, 1, memory_order_seq_cst, memory_scope_work_group); // expected-error {{address argument to atomic operation must be a pointer to atomic integer ('__generic atomic_float *' (aka '__generic _Atomic(float) *') invalid)}}

  bool cmpexch_1 = __opencl_atomic_compare_exchange_strong(i, I, 1, memory_order_seq_cst, memory_order_seq_cst, memory_scope_work_group);
  bool cmpexch_2 = __opencl_atomic_compare_exchange_strong(p, P, 1, memory_order_seq_cst, memory_order_seq_cst, memory_scope_work_group);
  bool cmpexch_3 = __opencl_atomic_compare_exchange_strong(d, I, 1, memory_order_seq_cst, memory_order_seq_cst, memory_scope_work_group); // expected-warning {{incompatible pointer types passing '__generic int *__private' to parameter of type '__generic float *'}}
  (void)__opencl_atomic_compare_exchange_strong(i, CI, 1, memory_order_seq_cst, memory_order_seq_cst, memory_scope_work_group); // expected-warning {{passing 'const __generic int *__private' to parameter of type '__generic int *' discards qualifiers}}

  bool cmpexchw_1 = __opencl_atomic_compare_exchange_weak(i, I, 1, memory_order_seq_cst, memory_order_seq_cst, memory_scope_work_group);
  bool cmpexchw_2 = __opencl_atomic_compare_exchange_weak(p, P, 1, memory_order_seq_cst, memory_order_seq_cst, memory_scope_work_group);
  bool cmpexchw_3 = __opencl_atomic_compare_exchange_weak(d, I, 1, memory_order_seq_cst, memory_order_seq_cst, memory_scope_work_group); // expected-warning {{incompatible pointer types passing '__generic int *__private' to parameter of type '__generic float *'}}
  (void)__opencl_atomic_compare_exchange_weak(i, CI, 1, memory_order_seq_cst, memory_order_seq_cst, memory_scope_work_group); // expected-warning {{passing 'const __generic int *__private' to parameter of type '__generic int *' discards qualifiers}}

  // Pointers to different address spaces are allowed.
  bool cmpexch_10 = __opencl_atomic_compare_exchange_strong((global atomic_int *)0x308, (constant int *)0x309, 1, memory_order_seq_cst, memory_order_seq_cst, memory_scope_work_group);

  __opencl_atomic_init(ci, 0); // expected-error {{address argument to atomic operation must be a pointer to non-const _Atomic type ('const __generic atomic_int *' (aka 'const __generic _Atomic(int) *') invalid)}}
  __opencl_atomic_store(ci, 0, memory_order_release, memory_scope_work_group); // expected-error {{address argument to atomic operation must be a pointer to non-const _Atomic type ('const __generic atomic_int *' (aka 'const __generic _Atomic(int) *') invalid)}}
  __opencl_atomic_load(ci, memory_order_acquire, memory_scope_work_group);

  __opencl_atomic_init(&gn, 456);
  __opencl_atomic_init(&gn, (void*)0); // expected-warning{{incompatible pointer to integer conversion passing '__generic void *' to parameter of type 'int'}}
}

void memory_checks(atomic_int *Ap, int *p, int val) {
  // non-integer memory order argument is casted to integer type.
  (void)__opencl_atomic_load(Ap, 1.0f, memory_scope_work_group);
  float forder;
  (void)__opencl_atomic_load(Ap, forder, memory_scope_work_group);
  struct S s;
  (void)__opencl_atomic_load(Ap, s, memory_scope_work_group); // expected-error {{passing '__private struct S' to parameter of incompatible type 'int'}}

  (void)__opencl_atomic_load(Ap, memory_order_relaxed, memory_scope_work_group);
  (void)__opencl_atomic_load(Ap, memory_order_acquire, memory_scope_work_group);
  (void)__opencl_atomic_load(Ap, memory_order_consume, memory_scope_work_group); // expected-error {{use of undeclared identifier 'memory_order_consume'}}
  (void)__opencl_atomic_load(Ap, memory_order_release, memory_scope_work_group); // expected-warning {{memory order argument to atomic operation is invalid}}
  (void)__opencl_atomic_load(Ap, memory_order_acq_rel, memory_scope_work_group); // expected-warning {{memory order argument to atomic operation is invalid}}
  (void)__opencl_atomic_load(Ap, memory_order_seq_cst, memory_scope_work_group);

  (void)__opencl_atomic_store(Ap, val, memory_order_relaxed, memory_scope_work_group);
  (void)__opencl_atomic_store(Ap, val, memory_order_acquire, memory_scope_work_group); // expected-warning {{memory order argument to atomic operation is invalid}}
  (void)__opencl_atomic_store(Ap, val, memory_order_release, memory_scope_work_group);
  (void)__opencl_atomic_store(Ap, val, memory_order_acq_rel, memory_scope_work_group); // expected-warning {{memory order argument to atomic operation is invalid}}
  (void)__opencl_atomic_store(Ap, val, memory_order_seq_cst, memory_scope_work_group);

  (void)__opencl_atomic_fetch_add(Ap, 1, memory_order_relaxed, memory_scope_work_group);
  (void)__opencl_atomic_fetch_add(Ap, 1, memory_order_acquire, memory_scope_work_group);
  (void)__opencl_atomic_fetch_add(Ap, 1, memory_order_release, memory_scope_work_group);
  (void)__opencl_atomic_fetch_add(Ap, 1, memory_order_acq_rel, memory_scope_work_group);
  (void)__opencl_atomic_fetch_add(Ap, 1, memory_order_seq_cst, memory_scope_work_group);

  (void)__opencl_atomic_init(Ap, val);

  (void)__opencl_atomic_fetch_sub(Ap, val, memory_order_relaxed, memory_scope_work_group);
  (void)__opencl_atomic_fetch_sub(Ap, val, memory_order_acquire, memory_scope_work_group);
  (void)__opencl_atomic_fetch_sub(Ap, val, memory_order_release, memory_scope_work_group);
  (void)__opencl_atomic_fetch_sub(Ap, val, memory_order_acq_rel, memory_scope_work_group);
  (void)__opencl_atomic_fetch_sub(Ap, val, memory_order_seq_cst, memory_scope_work_group);

  (void)__opencl_atomic_fetch_and(Ap, val, memory_order_relaxed, memory_scope_work_group);
  (void)__opencl_atomic_fetch_and(Ap, val, memory_order_acquire, memory_scope_work_group);
  (void)__opencl_atomic_fetch_and(Ap, val, memory_order_release, memory_scope_work_group);
  (void)__opencl_atomic_fetch_and(Ap, val, memory_order_acq_rel, memory_scope_work_group);
  (void)__opencl_atomic_fetch_and(Ap, val, memory_order_seq_cst, memory_scope_work_group);

  (void)__opencl_atomic_fetch_or(Ap, val, memory_order_relaxed, memory_scope_work_group);
  (void)__opencl_atomic_fetch_or(Ap, val, memory_order_acquire, memory_scope_work_group);
  (void)__opencl_atomic_fetch_or(Ap, val, memory_order_release, memory_scope_work_group);
  (void)__opencl_atomic_fetch_or(Ap, val, memory_order_acq_rel, memory_scope_work_group);
  (void)__opencl_atomic_fetch_or(Ap, val, memory_order_seq_cst, memory_scope_work_group);

  (void)__opencl_atomic_fetch_xor(Ap, val, memory_order_relaxed, memory_scope_work_group);
  (void)__opencl_atomic_fetch_xor(Ap, val, memory_order_acquire, memory_scope_work_group);
  (void)__opencl_atomic_fetch_xor(Ap, val, memory_order_release, memory_scope_work_group);
  (void)__opencl_atomic_fetch_xor(Ap, val, memory_order_acq_rel, memory_scope_work_group);
  (void)__opencl_atomic_fetch_xor(Ap, val, memory_order_seq_cst, memory_scope_work_group);

  (void)__opencl_atomic_exchange(Ap, val, memory_order_relaxed, memory_scope_work_group);
  (void)__opencl_atomic_exchange(Ap, val, memory_order_acquire, memory_scope_work_group);
  (void)__opencl_atomic_exchange(Ap, val, memory_order_release, memory_scope_work_group);
  (void)__opencl_atomic_exchange(Ap, val, memory_order_acq_rel, memory_scope_work_group);
  (void)__opencl_atomic_exchange(Ap, val, memory_order_seq_cst, memory_scope_work_group);

  (void)__opencl_atomic_compare_exchange_strong(Ap, p, val, memory_order_relaxed, memory_order_relaxed, memory_scope_work_group);
  (void)__opencl_atomic_compare_exchange_strong(Ap, p, val, memory_order_acquire, memory_order_relaxed, memory_scope_work_group);
  (void)__opencl_atomic_compare_exchange_strong(Ap, p, val, memory_order_release, memory_order_relaxed, memory_scope_work_group);
  (void)__opencl_atomic_compare_exchange_strong(Ap, p, val, memory_order_acq_rel, memory_order_relaxed, memory_scope_work_group);
  (void)__opencl_atomic_compare_exchange_strong(Ap, p, val, memory_order_seq_cst, memory_order_relaxed, memory_scope_work_group);

  (void)__opencl_atomic_compare_exchange_weak(Ap, p, val, memory_order_relaxed, memory_order_relaxed, memory_scope_work_group);
  (void)__opencl_atomic_compare_exchange_weak(Ap, p, val, memory_order_acquire, memory_order_relaxed, memory_scope_work_group);
  (void)__opencl_atomic_compare_exchange_weak(Ap, p, val, memory_order_release, memory_order_relaxed, memory_scope_work_group);
  (void)__opencl_atomic_compare_exchange_weak(Ap, p, val, memory_order_acq_rel, memory_order_relaxed, memory_scope_work_group);
  (void)__opencl_atomic_compare_exchange_weak(Ap, p, val, memory_order_seq_cst, memory_order_relaxed, memory_scope_work_group);
}

void synchscope_checks(atomic_int *Ap, int scope) {
  (void)__opencl_atomic_load(Ap, memory_order_relaxed, memory_scope_work_item); // expected-error{{synchronization scope argument to atomic operation is invalid}}
  (void)__opencl_atomic_load(Ap, memory_order_relaxed, memory_scope_work_group);
  (void)__opencl_atomic_load(Ap, memory_order_relaxed, memory_scope_device);
  (void)__opencl_atomic_load(Ap, memory_order_relaxed, memory_scope_all_svm_devices);
  (void)__opencl_atomic_load(Ap, memory_order_relaxed, memory_scope_sub_group);
  (void)__opencl_atomic_load(Ap, memory_order_relaxed, scope);
  (void)__opencl_atomic_load(Ap, memory_order_relaxed, 10);    //expected-error{{synchronization scope argument to atomic operation is invalid}}

  // non-integer memory scope is casted to integer type.
  float fscope;
  (void)__opencl_atomic_load(Ap, memory_order_relaxed, 1.0f);
  (void)__opencl_atomic_load(Ap, memory_order_relaxed, fscope);
  struct S s;
  (void)__opencl_atomic_load(Ap, memory_order_relaxed, s); //expected-error{{passing '__private struct S' to parameter of incompatible type 'int'}}
}

void nullPointerWarning(atomic_int *Ap, int *p, int val) {
  // The 'expected' pointer shouldn't be NULL.
  (void)__opencl_atomic_compare_exchange_strong(Ap, (void *)0, val, memory_order_relaxed, memory_order_relaxed, memory_scope_work_group); // expected-warning {{null passed to a callee that requires a non-null argument}}
}
