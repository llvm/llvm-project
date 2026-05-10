// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -fsanitize=null,alignment,array-bounds -std=c11 -O0 %s -o %t.c.ll && FileCheck %s --check-prefixes=C,SHARED < %t.c.ll
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -fsanitize=null,alignment,array-bounds -std=c++17 -x c++ -O0 %s -o %t.cxx.ll && FileCheck %s --check-prefixes=CXX,SHARED < %t.cxx.ll

// Precommit test for null, alignment, and array-bounds checks on aggregates.
// This test documents current behavior: memcpy is called but source operand is not checked
// for null/alignment (unlike scalar types). Array bounds checks exist for local
// arrays but not for past-the-end pointer accesses via parameters.

struct Small { int x; };
struct Container { struct Small inner; };

#ifdef __cplusplus
extern "C" {
#endif

// Plain type - arr[idx] operand form (known bounds)

// SHARED-LABEL: define {{[^@]*}}@test_assign_plain_arr_idx
// SHARED: [[ARR:%.*]] = load ptr, ptr %arr.addr
// SHARED: [[SRC:%.*]] = getelementptr inbounds %struct.Small, ptr [[ARR]], i64 0
// SHARED-NOT: call void @__ubsan_handle_type_mismatch_v1_abort
// SHARED-NOT: icmp ne ptr [[SRC]], null
// SHARED: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %{{.*}}, ptr align 4 [[SRC]], i64 4, i1 false)
__attribute__((noinline)) void test_assign_plain_arr_idx(struct Small *dest, struct Small arr[4]) {
  *dest = arr[0];
}

// SHARED-LABEL: define {{[^@]*}}@test_init_plain_arr_idx
// SHARED: [[ARR:%.*]] = load ptr, ptr %arr.addr
// SHARED: [[SRC:%.*]] = getelementptr inbounds %struct.Small, ptr [[ARR]], i64 0
// SHARED-NOT: call void @__ubsan_handle_type_mismatch_v1_abort
// SHARED-NOT: icmp ne ptr [[SRC]], null
// SHARED: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %{{.*}}, ptr align 4 [[SRC]], i64 4, i1 false)
__attribute__((noinline)) void test_init_plain_arr_idx(struct Small arr[4]) {
  struct Small a = arr[0];
}

// SHARED-LABEL: define {{[^@]*}}@test_init_list_plain_arr_idx
// SHARED: [[ARR:%.*]] = load ptr, ptr %arr.addr
// SHARED: [[SRC:%.*]] = getelementptr inbounds %struct.Small, ptr [[ARR]], i64 0
// SHARED-NOT: call void @__ubsan_handle_type_mismatch_v1_abort
// SHARED-NOT: icmp ne ptr [[SRC]], null
// SHARED: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %{{.*}}, ptr align 4 [[SRC]], i64 4, i1 false)
__attribute__((noinline)) void test_init_list_plain_arr_idx(struct Small arr[4]) {
  struct Small a[] = {arr[0]};
}

// SHARED-LABEL: define {{[^@]*}}@test_nested_member_plain_arr_idx
// SHARED: call void @__ubsan_handle_type_mismatch_v1_abort
// SHARED: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_nested_member_plain_arr_idx(struct Container *c, struct Small arr[4]) {
  c->inner = arr[0];
}

// Plain type - *ap operand form

// SHARED-LABEL: define {{[^@]*}}@test_assign_plain_deref_ptr
// SHARED: [[SRC:%.*]] = load ptr, ptr %ap.addr
// SHARED-NOT: call void @__ubsan_handle_type_mismatch_v1_abort
// SHARED-NOT: icmp ne ptr [[SRC]], null
// SHARED: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %{{.*}}, ptr align 4 [[SRC]], i64 4, i1 false)
__attribute__((noinline)) void test_assign_plain_deref_ptr(struct Small *dest, struct Small *ap) {
  *dest = *ap;
}

// SHARED-LABEL: define {{[^@]*}}@test_init_plain_deref_ptr
// SHARED: [[SRC:%.*]] = load ptr, ptr %ap.addr
// SHARED-NOT: call void @__ubsan_handle_type_mismatch_v1_abort
// SHARED-NOT: icmp ne ptr [[SRC]], null
// SHARED: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %{{.*}}, ptr align 4 [[SRC]], i64 4, i1 false)
__attribute__((noinline)) void test_init_plain_deref_ptr(struct Small *ap) {
  struct Small a = *ap;
}

// SHARED-LABEL: define {{[^@]*}}@test_init_list_plain_deref_ptr
// SHARED: [[SRC:%.*]] = load ptr, ptr %ap.addr
// SHARED-NOT: call void @__ubsan_handle_type_mismatch_v1_abort
// SHARED-NOT: icmp ne ptr [[SRC]], null
// SHARED: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %{{.*}}, ptr align 4 [[SRC]], i64 4, i1 false)
__attribute__((noinline)) void test_init_list_plain_deref_ptr(struct Small *ap) {
  struct Small a[] = {*ap};
}

// SHARED-LABEL: define {{[^@]*}}@test_nested_member_plain_deref_ptr
// SHARED: call void @__ubsan_handle_type_mismatch_v1_abort
// SHARED: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_nested_member_plain_deref_ptr(struct Container *c, struct Small *ap) {
  c->inner = *ap;
}

// Misaligned aggregate access

// SHARED-LABEL: define {{[^@]*}}@test_misaligned_access
// SHARED: %[[P:.*]] = load ptr, ptr %p
// SHARED-NOT: call void @__ubsan_handle_type_mismatch_v1_abort
// SHARED: call void @llvm.memcpy
__attribute__((noinline)) void test_misaligned_access(struct Small *dest, char *buf) {
  struct Small *p = (struct Small *)(buf + 1);  // Misaligned
  *dest = *p;
}

// Array bounds: out-of-bounds on local array

// SHARED-LABEL: define {{[^@]*}}@test_local_array_oob
// SHARED: call void @__ubsan_handle_out_of_bounds_abort
// SHARED-NOT: call void @__ubsan_handle_type_mismatch_v1_abort
// SHARED: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_local_array_oob(struct Small *dest) {
  struct Small arr[4];
  *dest = arr[5];
}

// Array bounds: past-the-end via parameter

// SHARED-LABEL: define {{[^@]*}}@test_past_the_end_arr_idx
// SHARED: [[ARR:%.*]] = load ptr, ptr %arr.addr
// SHARED: [[SRC:%.*]] = getelementptr inbounds %struct.Small, ptr [[ARR]], i64 4
// SHARED-NOT: call void @__ubsan_handle_out_of_bounds_abort
// SHARED-NOT: call void @__ubsan_handle_type_mismatch_v1_abort
// SHARED-NOT: icmp ne ptr [[SRC]], null
// SHARED: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %{{.*}}, ptr align 4 [[SRC]], i64 4, i1 false)
__attribute__((noinline)) void test_past_the_end_arr_idx(struct Small *dest, struct Small arr[4]) {
  *dest = arr[4];
}

// SHARED-LABEL: define {{[^@]*}}@test_past_the_end_init
// SHARED: [[ARR:%.*]] = load ptr, ptr %arr.addr
// SHARED: [[SRC:%.*]] = getelementptr inbounds %struct.Small, ptr [[ARR]], i64 4
// SHARED-NOT: call void @__ubsan_handle_out_of_bounds_abort
// SHARED-NOT: call void @__ubsan_handle_type_mismatch_v1_abort
// SHARED-NOT: icmp ne ptr [[SRC]], null
// SHARED: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %{{.*}}, ptr align 4 [[SRC]], i64 4, i1 false)
__attribute__((noinline)) void test_past_the_end_init(struct Small arr[4]) {
  struct Small a = arr[4];
}

#ifdef __cplusplus
} // extern "C"
#endif

// Atomic type (C only)

#ifndef __cplusplus

// C-LABEL: define {{[^@]*}}@test_assign_atomic_deref_ptr
// C: [[SRC:%.*]] = load ptr, ptr %ap.addr
// C-NOT: call void @__ubsan_handle_type_mismatch_v1_abort
// C-NOT: icmp ne ptr [[SRC]], null
// C: load atomic i32, ptr [[SRC]] seq_cst
__attribute__((noinline)) void test_assign_atomic_deref_ptr(struct Small *dest, _Atomic(struct Small) *ap) {
  *dest = *ap;
}

#endif // !__cplusplus

// C++ only

#ifdef __cplusplus

extern "C" {

// CXX-LABEL: define {{[^@]*}}@test_cxx_init_direct_plain_arr_idx
// CXX: [[ARR:%.*]] = load ptr, ptr %arr.addr
// CXX: [[SRC:%.*]] = getelementptr inbounds %struct.Small, ptr [[ARR]], i64 0
// CXX-NOT: call void @__ubsan_handle_type_mismatch_v1_abort
// CXX-NOT: icmp ne ptr [[SRC]], null
// CXX: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %{{.*}}, ptr align 4 [[SRC]], i64 4, i1 false)
__attribute__((noinline)) void test_cxx_init_direct_plain_arr_idx(struct Small arr[4]) {
  struct Small a(arr[0]);
}

// CXX-LABEL: define {{[^@]*}}@test_cxx_init_brace_plain_arr_idx
// CXX: [[ARR:%.*]] = load ptr, ptr %arr.addr
// CXX: [[SRC:%.*]] = getelementptr inbounds %struct.Small, ptr [[ARR]], i64 0
// CXX-NOT: call void @__ubsan_handle_type_mismatch_v1_abort
// CXX-NOT: icmp ne ptr [[SRC]], null
// CXX: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %{{.*}}, ptr align 4 [[SRC]], i64 4, i1 false)
__attribute__((noinline)) void test_cxx_init_brace_plain_arr_idx(struct Small arr[4]) {
  struct Small a{arr[0]};
}

// CXX-LABEL: define {{[^@]*}}@test_cxx_init_direct_plain_deref_ptr
// CXX: [[SRC:%.*]] = load ptr, ptr %ap.addr
// CXX-NOT: call void @__ubsan_handle_type_mismatch_v1_abort
// CXX-NOT: icmp ne ptr [[SRC]], null
// CXX: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %{{.*}}, ptr align 4 [[SRC]], i64 4, i1 false)
__attribute__((noinline)) void test_cxx_init_direct_plain_deref_ptr(struct Small *ap) {
  struct Small a(*ap);
}

// CXX-LABEL: define {{[^@]*}}@test_cxx_init_brace_plain_deref_ptr
// CXX: [[SRC:%.*]] = load ptr, ptr %ap.addr
// CXX-NOT: call void @__ubsan_handle_type_mismatch_v1_abort
// CXX-NOT: icmp ne ptr [[SRC]], null
// CXX: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %{{.*}}, ptr align 4 [[SRC]], i64 4, i1 false)
__attribute__((noinline)) void test_cxx_init_brace_plain_deref_ptr(struct Small *ap) {
  struct Small a{*ap};
}

// CXX-LABEL: define {{[^@]*}}@test_cxx_new_direct_plain_deref_ptr
// CXX: [[SRC:%.*]] = load ptr, ptr %ap.addr
// CXX-NOT: call void @__ubsan_handle_type_mismatch_v1_abort
// CXX-NOT: icmp ne ptr [[SRC]], null
// CXX: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %{{.*}}, ptr align 4 [[SRC]], i64 4, i1 false)
__attribute__((noinline)) void test_cxx_new_direct_plain_deref_ptr(struct Small *ap) {
  struct Small *a = new struct Small(*ap);
  delete a;
}

// C++ past-the-end tests

// CXX-LABEL: define {{[^@]*}}@test_cxx_past_the_end_direct
// CXX: [[ARR:%.*]] = load ptr, ptr %arr.addr
// CXX: [[SRC:%.*]] = getelementptr inbounds %struct.Small, ptr [[ARR]], i64 4
// CXX-NOT: call void @__ubsan_handle_out_of_bounds_abort
// CXX-NOT: call void @__ubsan_handle_type_mismatch_v1_abort
// CXX-NOT: icmp ne ptr [[SRC]], null
// CXX: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %{{.*}}, ptr align 4 [[SRC]], i64 4, i1 false)
__attribute__((noinline)) void test_cxx_past_the_end_direct(struct Small arr[4]) {
  struct Small a(arr[4]);
}

// CXX-LABEL: define {{[^@]*}}@test_cxx_past_the_end_brace
// CXX: [[ARR:%.*]] = load ptr, ptr %arr.addr
// CXX: [[SRC:%.*]] = getelementptr inbounds %struct.Small, ptr [[ARR]], i64 4
// CXX-NOT: call void @__ubsan_handle_out_of_bounds_abort
// CXX-NOT: call void @__ubsan_handle_type_mismatch_v1_abort
// CXX-NOT: icmp ne ptr [[SRC]], null
// CXX: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %{{.*}}, ptr align 4 [[SRC]], i64 4, i1 false)
__attribute__((noinline)) void test_cxx_past_the_end_brace(struct Small arr[4]) {
  struct Small a{arr[4]};
}

} // extern "C"

#endif // __cplusplus
