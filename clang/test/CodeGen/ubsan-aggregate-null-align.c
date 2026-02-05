// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -fsanitize=null,alignment,array-bounds -std=c11 -O0 %s -o %t.c.ll && FileCheck %s --check-prefixes=C,SHARED < %t.c.ll
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -fsanitize=null,alignment,array-bounds -std=c++17 -x c++ -O0 %s -o %t.cxx.ll && FileCheck %s --check-prefixes=CXX,SHARED < %t.cxx.ll

struct Small { int x; };
struct Container { struct Small inner; };
struct SmallWrapper { struct Small a; };

extern void variadic_func(int, ...);

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// TYPE VARIANT: plain
// ============================================================================

// --- OPERAND FORM: arr[idx] ---

// SHARED-LABEL: define {{[^@]*}}@test_assign_plain_arr_idx
// SHARED: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_assign_plain_arr_idx(struct Small *dest, struct Small arr[4]) {
  *dest = arr[0];
}

// SHARED-LABEL: define {{[^@]*}}@test_init_plain_arr_idx
// SHARED: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_init_plain_arr_idx(struct Small arr[4]) {
  struct Small a = arr[0];
}

// SHARED-LABEL: define {{[^@]*}}@test_init_list_plain_arr_idx
// SHARED: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_init_list_plain_arr_idx(struct Small arr[4]) {
  struct Small a[] = {arr[0]};
}

// SHARED-LABEL: define {{[^@]*}}@test_variadic_plain_arr_idx
__attribute__((noinline)) void test_variadic_plain_arr_idx(struct Small arr[4]) {
  variadic_func(0, arr[0]);
}

// SHARED-LABEL: define {{[^@]*}}@test_nested_member_plain_arr_idx
// SHARED: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_nested_member_plain_arr_idx(struct Container *c, struct Small arr[4]) {
  c->inner = arr[0];
}

// --- OPERAND FORM: *ap ---

// SHARED-LABEL: define {{[^@]*}}@test_assign_plain_deref_ptr
// SHARED: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_assign_plain_deref_ptr(struct Small *dest, struct Small *ap) {
  *dest = *ap;
}

// SHARED-LABEL: define {{[^@]*}}@test_init_plain_deref_ptr
// SHARED: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_init_plain_deref_ptr(struct Small *ap) {
  struct Small a = *ap;
}

// SHARED-LABEL: define {{[^@]*}}@test_init_list_plain_deref_ptr
// SHARED: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_init_list_plain_deref_ptr(struct Small *ap) {
  struct Small a[] = {*ap};
}

// SHARED-LABEL: define {{[^@]*}}@test_variadic_plain_deref_ptr
__attribute__((noinline)) void test_variadic_plain_deref_ptr(struct Small *ap) {
  variadic_func(0, *ap);
}

// SHARED-LABEL: define {{[^@]*}}@test_nested_member_plain_deref_ptr
// SHARED: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_nested_member_plain_deref_ptr(struct Container *c, struct Small *ap) {
  c->inner = *ap;
}

// ============================================================================
// ARRAY BOUNDS CHECKING
// Tests for array out-of-bounds access, including past-the-end pointer cases.
// Currently, bounds checking triggers for invalid indices but the planned fix
// will also trigger when indexing generates a past-the-end pointer.
// ============================================================================

// --- In-bounds access (index 0 of array[4]) - should NOT trigger bounds error ---

// SHARED-LABEL: define {{[^@]*}}@test_bounds_inbounds_idx0
// SHARED: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_bounds_inbounds_idx0(struct Small *dest, struct Small arr[4]) {
  *dest = arr[0];
}

// --- In-bounds access (index 3, last valid element of array[4]) ---

// SHARED-LABEL: define {{[^@]*}}@test_bounds_inbounds_idx3
// SHARED: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_bounds_inbounds_idx3(struct Small *dest, struct Small arr[4]) {
  *dest = arr[3];
}

// --- Out-of-bounds access (index 4, past-the-end of array[4]) ---
// This generates the past-the-end pointer. Currently no bounds check is emitted
// for this case; the planned fix will add checking for past-the-end access.

// SHARED-LABEL: define {{[^@]*}}@test_bounds_oob_past_end
// SHARED: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_bounds_oob_past_end(struct Small *dest, struct Small arr[4]) {
  *dest = arr[4];
}

// --- Out-of-bounds access (index 5, clearly beyond array[4]) ---

// SHARED-LABEL: define {{[^@]*}}@test_bounds_oob_beyond
// SHARED: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_bounds_oob_beyond(struct Small *dest, struct Small arr[4]) {
  *dest = arr[5];
}

// --- Dynamic index bounds checking ---

// SHARED-LABEL: define {{[^@]*}}@test_bounds_dynamic_idx
// SHARED: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_bounds_dynamic_idx(struct Small *dest, struct Small arr[4], int idx) {
  *dest = arr[idx];
}

// --- Bounds checking with initialization ---

// SHARED-LABEL: define {{[^@]*}}@test_bounds_init_past_end
// SHARED: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_bounds_init_past_end(struct Small arr[4]) {
  struct Small a = arr[4];
}

// --- Bounds checking with nested member assignment ---

// SHARED-LABEL: define {{[^@]*}}@test_bounds_nested_past_end
// SHARED: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_bounds_nested_past_end(struct Container *c, struct Small arr[4]) {
  c->inner = arr[4];
}

// --- Bounds checking in initializer list ---

// SHARED-LABEL: define {{[^@]*}}@test_bounds_init_list_past_end
// SHARED: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_bounds_init_list_past_end(struct Small arr[4]) {
  struct Small a[] = {arr[4]};
}

// --- Bounds checking with variadic function ---

// SHARED-LABEL: define {{[^@]*}}@test_bounds_variadic_past_end
__attribute__((noinline)) void test_bounds_variadic_past_end(struct Small arr[4]) {
  variadic_func(0, arr[4]);
}

#ifdef __cplusplus
} // extern "C"
#endif

// ============================================================================
// TYPE VARIANT: atomic (C only)
// ============================================================================

#ifndef __cplusplus

// --- OPERAND FORM: arr[idx] ---

// C-LABEL: define {{[^@]*}}@test_assign_atomic_arr_idx
// C: load atomic i32
__attribute__((noinline)) void test_assign_atomic_arr_idx(struct Small *dest, _Atomic(struct Small) arr[4]) {
  *dest = arr[0];
}

// C-LABEL: define {{[^@]*}}@test_init_atomic_arr_idx
// C: load atomic i32
__attribute__((noinline)) void test_init_atomic_arr_idx(_Atomic(struct Small) arr[4]) {
  struct Small a = arr[0];
}

// C-LABEL: define {{[^@]*}}@test_init_list_atomic_arr_idx
// C: load atomic i32
__attribute__((noinline)) void test_init_list_atomic_arr_idx(_Atomic(struct Small) arr[4]) {
  struct Small a[] = {arr[0]};
}

// C-LABEL: define {{[^@]*}}@test_init_list_designate_atomic_arr_idx
// C: load atomic i32
__attribute__((noinline)) void test_init_list_designate_atomic_arr_idx(_Atomic(struct Small) arr[4]) {
  struct Small a[] = {[0] = arr[0]};
}

// C-LABEL: define {{[^@]*}}@test_variadic_atomic_arr_idx
// C: load atomic i32
__attribute__((noinline)) void test_variadic_atomic_arr_idx(_Atomic(struct Small) arr[4]) {
  variadic_func(0, arr[0]);
}

// C-LABEL: define {{[^@]*}}@test_nested_member_atomic_arr_idx
// C: load atomic i32
__attribute__((noinline)) void test_nested_member_atomic_arr_idx(struct Container *c, _Atomic(struct Small) arr[4]) {
  c->inner = arr[0];
}

// C-LABEL: define {{[^@]*}}@test_lvalue_to_rvalue_atomic_arr_idx
// C: load atomic i32
__attribute__((noinline)) void test_lvalue_to_rvalue_atomic_arr_idx(_Atomic(struct Small) arr[4]) {
  (void)arr[0];
}

// --- OPERAND FORM: *ap ---

// C-LABEL: define {{[^@]*}}@test_assign_atomic_deref_ptr
// C: load atomic i32
__attribute__((noinline)) void test_assign_atomic_deref_ptr(struct Small *dest, _Atomic(struct Small) *ap) {
  *dest = *ap;
}

// C-LABEL: define {{[^@]*}}@test_init_atomic_deref_ptr
// C: load atomic i32
__attribute__((noinline)) void test_init_atomic_deref_ptr(_Atomic(struct Small) *ap) {
  struct Small a = *ap;
}

// C-LABEL: define {{[^@]*}}@test_init_list_atomic_deref_ptr
// C: load atomic i32
__attribute__((noinline)) void test_init_list_atomic_deref_ptr(_Atomic(struct Small) *ap) {
  struct Small a[] = {*ap};
}

// C-LABEL: define {{[^@]*}}@test_init_list_designate_atomic_deref_ptr
// C: load atomic i32
__attribute__((noinline)) void test_init_list_designate_atomic_deref_ptr(_Atomic(struct Small) *ap) {
  struct Small a[] = {[0] = *ap};
}

// C-LABEL: define {{[^@]*}}@test_variadic_atomic_deref_ptr
// C: load atomic i32
__attribute__((noinline)) void test_variadic_atomic_deref_ptr(_Atomic(struct Small) *ap) {
  variadic_func(0, *ap);
}

// C-LABEL: define {{[^@]*}}@test_nested_member_atomic_deref_ptr
// C: load atomic i32
__attribute__((noinline)) void test_nested_member_atomic_deref_ptr(struct Container *c, _Atomic(struct Small) *ap) {
  c->inner = *ap;
}

// C-LABEL: define {{[^@]*}}@test_lvalue_to_rvalue_atomic_deref_ptr
// C: load atomic i32
__attribute__((noinline)) void test_lvalue_to_rvalue_atomic_deref_ptr(_Atomic(struct Small) *ap) {
  (void)*ap;
}

// ============================================================================
// TYPE VARIANT: volatile (C only)
// ============================================================================

// --- OPERAND FORM: arr[idx] ---

// C-LABEL: define {{[^@]*}}@test_assign_volatile_arr_idx
// C: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_assign_volatile_arr_idx(struct Small *dest, volatile struct Small arr[4]) {
  *dest = arr[0];
}

// C-LABEL: define {{[^@]*}}@test_init_volatile_arr_idx
// C: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_init_volatile_arr_idx(volatile struct Small arr[4]) {
  struct Small a = arr[0];
}

// C-LABEL: define {{[^@]*}}@test_init_list_volatile_arr_idx
// C: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_init_list_volatile_arr_idx(volatile struct Small arr[4]) {
  struct Small a[] = {arr[0]};
}

// C-LABEL: define {{[^@]*}}@test_init_list_designate_volatile_arr_idx
// C: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_init_list_designate_volatile_arr_idx(volatile struct Small arr[4]) {
  struct Small a[] = {[0] = arr[0]};
}

// C-LABEL: define {{[^@]*}}@test_variadic_volatile_arr_idx
__attribute__((noinline)) void test_variadic_volatile_arr_idx(volatile struct Small arr[4]) {
  variadic_func(0, arr[0]);
}

// C-LABEL: define {{[^@]*}}@test_nested_member_volatile_arr_idx
// C: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_nested_member_volatile_arr_idx(struct Container *c, volatile struct Small arr[4]) {
  c->inner = arr[0];
}

// C-LABEL: define {{[^@]*}}@test_lvalue_to_rvalue_volatile_arr_idx
// C: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_lvalue_to_rvalue_volatile_arr_idx(volatile struct Small arr[4]) {
  (void)arr[0];
}

// --- OPERAND FORM: *ap ---

// C-LABEL: define {{[^@]*}}@test_assign_volatile_deref_ptr
// C: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_assign_volatile_deref_ptr(struct Small *dest, volatile struct Small *ap) {
  *dest = *ap;
}

// C-LABEL: define {{[^@]*}}@test_init_volatile_deref_ptr
// C: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_init_volatile_deref_ptr(volatile struct Small *ap) {
  struct Small a = *ap;
}

// C-LABEL: define {{[^@]*}}@test_init_list_volatile_deref_ptr
// C: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_init_list_volatile_deref_ptr(volatile struct Small *ap) {
  struct Small a[] = {*ap};
}

// C-LABEL: define {{[^@]*}}@test_init_list_designate_volatile_deref_ptr
// C: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_init_list_designate_volatile_deref_ptr(volatile struct Small *ap) {
  struct Small a[] = {[0] = *ap};
}

// C-LABEL: define {{[^@]*}}@test_variadic_volatile_deref_ptr
__attribute__((noinline)) void test_variadic_volatile_deref_ptr(volatile struct Small *ap) {
  variadic_func(0, *ap);
}

// C-LABEL: define {{[^@]*}}@test_nested_member_volatile_deref_ptr
// C: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_nested_member_volatile_deref_ptr(struct Container *c, volatile struct Small *ap) {
  c->inner = *ap;
}

// C-LABEL: define {{[^@]*}}@test_lvalue_to_rvalue_volatile_deref_ptr
// C: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_lvalue_to_rvalue_volatile_deref_ptr(volatile struct Small *ap) {
  (void)*ap;
}

// --- Designated initializers (C only) ---

// C-LABEL: define {{[^@]*}}@test_init_list_designate_plain_arr_idx
// C: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_init_list_designate_plain_arr_idx(struct Small arr[4]) {
  struct Small a[] = {[0] = arr[0]};
}

// C-LABEL: define {{[^@]*}}@test_init_list_designate_plain_deref_ptr
// C: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_init_list_designate_plain_deref_ptr(struct Small *ap) {
  struct Small a[] = {[0] = *ap};
}

#endif // !__cplusplus

// ============================================================================
// C++ ONLY: Additional initialization and operation forms
// ============================================================================

#ifdef __cplusplus

extern "C" {

// --- OPERAND FORM: arr[idx] with plain type ---

// CXX-LABEL: define {{[^@]*}}@test_cxx_init_direct_plain_arr_idx
// CXX: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_cxx_init_direct_plain_arr_idx(struct Small arr[4]) {
  struct Small a(arr[0]);
}

// CXX-LABEL: define {{[^@]*}}@test_cxx_init_brace_plain_arr_idx
// CXX: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_cxx_init_brace_plain_arr_idx(struct Small arr[4]) {
  struct Small a{arr[0]};
}

// CXX-LABEL: define {{[^@]*}}@test_cxx_init_copy_list_plain_arr_idx
// CXX: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_cxx_init_copy_list_plain_arr_idx(struct Small arr[4]) {
  struct Small a = {arr[0]};
}

// CXX-LABEL: define {{[^@]*}}@test_cxx_new_direct_plain_arr_idx
// CXX: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_cxx_new_direct_plain_arr_idx(struct Small arr[4]) {
  struct Small *a = new struct Small(arr[0]);
  delete a;
}

// CXX-LABEL: define {{[^@]*}}@test_cxx_new_brace_plain_arr_idx
// CXX: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_cxx_new_brace_plain_arr_idx(struct Small arr[4]) {
  struct Small *a = new struct Small{arr[0]};
  delete a;
}

// CXX-LABEL: define {{[^@]*}}@test_cxx_functional_cast_plain_arr_idx
// CXX: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_cxx_functional_cast_plain_arr_idx(struct Small arr[4]) {
  (void)Small(arr[0]);
}

// CXX-LABEL: define {{[^@]*}}@test_cxx_functional_cast_brace_plain_arr_idx
// CXX: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_cxx_functional_cast_brace_plain_arr_idx(struct Small arr[4]) {
  (void)Small{arr[0]};
}

// CXX-LABEL: define {{[^@]*}}@test_cxx_static_cast_plain_arr_idx
// CXX: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_cxx_static_cast_plain_arr_idx(struct Small arr[4]) {
  (void)static_cast<struct Small>(arr[0]);
}

// CXX-LABEL: define {{[^@]*}}@test_cxx_c_cast_plain_arr_idx
// CXX: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_cxx_c_cast_plain_arr_idx(struct Small arr[4]) {
  (void)(struct Small)(arr[0]);
}

// CXX-LABEL: define {{[^@]*}}@test_cxx_member_init_plain_arr_idx
// CXX: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_cxx_member_init_plain_arr_idx(struct Small arr[4]) {
  (void)new SmallWrapper{arr[0]};
}

// --- OPERAND FORM: *ap with plain type ---

// CXX-LABEL: define {{[^@]*}}@test_cxx_init_direct_plain_deref_ptr
// CXX: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_cxx_init_direct_plain_deref_ptr(struct Small *ap) {
  struct Small a(*ap);
}

// CXX-LABEL: define {{[^@]*}}@test_cxx_init_brace_plain_deref_ptr
// CXX: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_cxx_init_brace_plain_deref_ptr(struct Small *ap) {
  struct Small a{*ap};
}

// CXX-LABEL: define {{[^@]*}}@test_cxx_init_copy_list_plain_deref_ptr
// CXX: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_cxx_init_copy_list_plain_deref_ptr(struct Small *ap) {
  struct Small a = {*ap};
}

// CXX-LABEL: define {{[^@]*}}@test_cxx_new_direct_plain_deref_ptr
// CXX: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_cxx_new_direct_plain_deref_ptr(struct Small *ap) {
  struct Small *a = new struct Small(*ap);
  delete a;
}

// CXX-LABEL: define {{[^@]*}}@test_cxx_new_brace_plain_deref_ptr
// CXX: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_cxx_new_brace_plain_deref_ptr(struct Small *ap) {
  struct Small *a = new struct Small{*ap};
  delete a;
}

// CXX-LABEL: define {{[^@]*}}@test_cxx_functional_cast_plain_deref_ptr
// CXX: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_cxx_functional_cast_plain_deref_ptr(struct Small *ap) {
  (void)Small(*ap);
}

// CXX-LABEL: define {{[^@]*}}@test_cxx_functional_cast_brace_plain_deref_ptr
// CXX: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_cxx_functional_cast_brace_plain_deref_ptr(struct Small *ap) {
  (void)Small{*ap};
}

// CXX-LABEL: define {{[^@]*}}@test_cxx_static_cast_plain_deref_ptr
// CXX: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_cxx_static_cast_plain_deref_ptr(struct Small *ap) {
  (void)static_cast<struct Small>(*ap);
}

// CXX-LABEL: define {{[^@]*}}@test_cxx_c_cast_plain_deref_ptr
// CXX: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_cxx_c_cast_plain_deref_ptr(struct Small *ap) {
  (void)(struct Small)(*ap);
}

// CXX-LABEL: define {{[^@]*}}@test_cxx_member_init_plain_deref_ptr
// CXX: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_cxx_member_init_plain_deref_ptr(struct Small *ap) {
  (void)new SmallWrapper{*ap};
}

// --- VARIADIC with plain type ---

// CXX-LABEL: define {{[^@]*}}@test_cxx_variadic_plain_arr_idx
__attribute__((noinline)) void test_cxx_variadic_plain_arr_idx(struct Small arr[4]) {
  variadic_func(0, arr[0]);
}

// CXX-LABEL: define {{[^@]*}}@test_cxx_variadic_plain_deref_ptr
__attribute__((noinline)) void test_cxx_variadic_plain_deref_ptr(struct Small *ap) {
  variadic_func(0, *ap);
}

// --- Explicit operator= calls ---

// CXX-LABEL: define {{[^@]*}}@test_cxx_operator_assign_lhs_arr_idx
// CXX: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_cxx_operator_assign_lhs_arr_idx(struct Small *dest, struct Small arr[4]) {
  dest->operator=(arr[0]);
}

// CXX-LABEL: define {{[^@]*}}@test_cxx_operator_assign_lhs_deref_ptr
// CXX: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_cxx_operator_assign_lhs_deref_ptr(struct Small *dest, struct Small *ap) {
  dest->operator=(*ap);
}

// CXX-LABEL: define {{[^@]*}}@test_cxx_operator_assign_obj_arr_idx
// CXX: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_cxx_operator_assign_obj_arr_idx(struct Small arr[4]) {
  struct Small a;
  a.operator=(arr[0]);
}

// CXX-LABEL: define {{[^@]*}}@test_cxx_operator_assign_obj_deref_ptr
// CXX: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_cxx_operator_assign_obj_deref_ptr(struct Small *ap) {
  struct Small a;
  a.operator=(*ap);
}

// --- C++ bounds checking with past-the-end access ---

// CXX-LABEL: define {{[^@]*}}@test_cxx_bounds_init_direct_past_end
// CXX: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_cxx_bounds_init_direct_past_end(struct Small arr[4]) {
  struct Small a(arr[4]);
}

// CXX-LABEL: define {{[^@]*}}@test_cxx_bounds_init_brace_past_end
// CXX: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_cxx_bounds_init_brace_past_end(struct Small arr[4]) {
  struct Small a{arr[4]};
}

// CXX-LABEL: define {{[^@]*}}@test_cxx_bounds_new_past_end
// CXX: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_cxx_bounds_new_past_end(struct Small arr[4]) {
  struct Small *a = new struct Small(arr[4]);
  delete a;
}

} // extern "C"

// --- Virtual base initialization (cannot be extern "C") ---
// Use a simple struct to ensure memcpy is used for the copy

struct VirtualBaseSimple {
  int x;
};

struct DerivedVirtualArrSimple : virtual VirtualBaseSimple {
  __attribute__((noinline)) DerivedVirtualArrSimple(VirtualBaseSimple arr[4]) : VirtualBaseSimple(arr[0]) {}
};

struct DerivedVirtualPtrSimple : virtual VirtualBaseSimple {
  __attribute__((noinline)) DerivedVirtualPtrSimple(VirtualBaseSimple *ap) : VirtualBaseSimple(*ap) {}
};

// Force instantiation of constructors
// The constructors are emitted inline after their first use, so we check them
// in the order they appear: ArrSimple constructor, then wrapper for Ptr, then PtrSimple constructor

extern "C" {

// CXX-LABEL: define {{[^@]*}}@test_cxx_virtual_base_init_arr_idx
__attribute__((noinline)) void test_cxx_virtual_base_init_arr_idx(VirtualBaseSimple arr[4]) {
  DerivedVirtualArrSimple d(arr);
}

}

// Check the first constructor (DerivedVirtualArrSimple) - emitted after test_cxx_virtual_base_init_arr_idx
// CXX-LABEL: define {{[^@]*}}@_ZN23DerivedVirtualArrSimpleC1EP17VirtualBaseSimple
// CXX: call void @llvm.memcpy.p0.p0.i64

extern "C" {

// CXX-LABEL: define {{[^@]*}}@test_cxx_virtual_base_init_deref_ptr
__attribute__((noinline)) void test_cxx_virtual_base_init_deref_ptr(VirtualBaseSimple *ap) {
  DerivedVirtualPtrSimple d(ap);
}

}

// Check the second constructor (DerivedVirtualPtrSimple) - emitted after test_cxx_virtual_base_init_deref_ptr
// CXX-LABEL: define {{[^@]*}}@_ZN23DerivedVirtualPtrSimpleC1EP17VirtualBaseSimple
// CXX: call void @llvm.memcpy.p0.p0.i64

#endif // __cplusplus
