// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -fsanitize=null,alignment -std=c11 -O0 %s -o %t.c.ll && FileCheck %s --check-prefixes=C,SHARED < %t.c.ll
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -fsanitize=null,alignment -std=c++17 -x c++ -O0 %s -o %t.cxx.ll && FileCheck %s --check-prefixes=CXX,SHARED < %t.cxx.ll

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
__attribute__((noinline)) void test_assign_plain_arr_idx(struct Small *dest, struct Small arr[]) {
  *dest = arr[0];
}

// SHARED-LABEL: define {{[^@]*}}@test_init_plain_arr_idx
// SHARED: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_init_plain_arr_idx(struct Small arr[]) {
  struct Small a = arr[0];
}

// SHARED-LABEL: define {{[^@]*}}@test_init_list_plain_arr_idx
// SHARED: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_init_list_plain_arr_idx(struct Small arr[]) {
  struct Small a[] = {arr[0]};
}

// SHARED-LABEL: define {{[^@]*}}@test_init_list_designate_plain_arr_idx
// SHARED: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_init_list_designate_plain_arr_idx(struct Small arr[]) {
  struct Small a[] = {[0] = arr[0]};
}

// SHARED-LABEL: define {{[^@]*}}@test_variadic_plain_arr_idx
__attribute__((noinline)) void test_variadic_plain_arr_idx(struct Small arr[]) {
  variadic_func(0, arr[0]);
}

// SHARED-LABEL: define {{[^@]*}}@test_nested_member_plain_arr_idx
// SHARED: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_nested_member_plain_arr_idx(struct Container *c, struct Small arr[]) {
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

// SHARED-LABEL: define {{[^@]*}}@test_init_list_designate_plain_deref_ptr
// SHARED: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_init_list_designate_plain_deref_ptr(struct Small *ap) {
  struct Small a[] = {[0] = *ap};
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

#ifdef __cplusplus
} // extern "C"
#endif

// ============================================================================
// TYPE VARIANT: atomic (C only)
// NOTE: Atomic struct operations use atomic load/store instructions and do not
// go through EmitAggregateCopy, so UBSAN null/alignment checks are not emitted.
// ============================================================================

#ifndef __cplusplus

// --- OPERAND FORM: arr[idx] ---

// C-LABEL: define {{[^@]*}}@test_assign_atomic_arr_idx
// C: load atomic i32
__attribute__((noinline)) void test_assign_atomic_arr_idx(struct Small *dest, _Atomic(struct Small) arr[]) {
  *dest = arr[0];
}

// C-LABEL: define {{[^@]*}}@test_init_atomic_arr_idx
// C: load atomic i32
__attribute__((noinline)) void test_init_atomic_arr_idx(_Atomic(struct Small) arr[]) {
  struct Small a = arr[0];
}

// C-LABEL: define {{[^@]*}}@test_init_list_atomic_arr_idx
// C: load atomic i32
__attribute__((noinline)) void test_init_list_atomic_arr_idx(_Atomic(struct Small) arr[]) {
  struct Small a[] = {arr[0]};
}

// C-LABEL: define {{[^@]*}}@test_init_list_designate_atomic_arr_idx
// C: load atomic i32
__attribute__((noinline)) void test_init_list_designate_atomic_arr_idx(_Atomic(struct Small) arr[]) {
  struct Small a[] = {[0] = arr[0]};
}

// C-LABEL: define {{[^@]*}}@test_variadic_atomic_arr_idx
// C: load atomic i32
__attribute__((noinline)) void test_variadic_atomic_arr_idx(_Atomic(struct Small) arr[]) {
  variadic_func(0, arr[0]);
}

// C-LABEL: define {{[^@]*}}@test_nested_member_atomic_arr_idx
// C: load atomic i32
__attribute__((noinline)) void test_nested_member_atomic_arr_idx(struct Container *c, _Atomic(struct Small) arr[]) {
  c->inner = arr[0];
}

// C-LABEL: define {{[^@]*}}@test_lvalue_to_rvalue_atomic_arr_idx
// C: load atomic i32
__attribute__((noinline)) void test_lvalue_to_rvalue_atomic_arr_idx(_Atomic(struct Small) arr[]) {
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

#endif // !__cplusplus

// ============================================================================
// TYPE VARIANT: volatile (C only)
// ============================================================================

#ifndef __cplusplus

// --- OPERAND FORM: arr[idx] ---

// C-LABEL: define {{[^@]*}}@test_assign_volatile_arr_idx
// C: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_assign_volatile_arr_idx(struct Small *dest, volatile struct Small arr[]) {
  *dest = arr[0];
}

// C-LABEL: define {{[^@]*}}@test_init_volatile_arr_idx
// C: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_init_volatile_arr_idx(volatile struct Small arr[]) {
  struct Small a = arr[0];
}

// C-LABEL: define {{[^@]*}}@test_init_list_volatile_arr_idx
// C: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_init_list_volatile_arr_idx(volatile struct Small arr[]) {
  struct Small a[] = {arr[0]};
}

// C-LABEL: define {{[^@]*}}@test_init_list_designate_volatile_arr_idx
// C: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_init_list_designate_volatile_arr_idx(volatile struct Small arr[]) {
  struct Small a[] = {[0] = arr[0]};
}

// C-LABEL: define {{[^@]*}}@test_variadic_volatile_arr_idx
__attribute__((noinline)) void test_variadic_volatile_arr_idx(volatile struct Small arr[]) {
  variadic_func(0, arr[0]);
}

// C-LABEL: define {{[^@]*}}@test_nested_member_volatile_arr_idx
// C: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_nested_member_volatile_arr_idx(struct Container *c, volatile struct Small arr[]) {
  c->inner = arr[0];
}

// C-LABEL: define {{[^@]*}}@test_lvalue_to_rvalue_volatile_arr_idx
// C: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_lvalue_to_rvalue_volatile_arr_idx(volatile struct Small arr[]) {
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

#endif // !__cplusplus

// ============================================================================
// C++ ONLY: Additional initialization and operation forms
// ============================================================================

#ifdef __cplusplus

extern "C" {

// --- OPERAND FORM: arr[idx] with plain type ---

// CXX-LABEL: define {{[^@]*}}@test_cxx_init_direct_plain_arr_idx
// CXX: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_cxx_init_direct_plain_arr_idx(struct Small arr[]) {
  struct Small a(arr[0]);
}

// CXX-LABEL: define {{[^@]*}}@test_cxx_init_brace_plain_arr_idx
// CXX: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_cxx_init_brace_plain_arr_idx(struct Small arr[]) {
  struct Small a{arr[0]};
}

// CXX-LABEL: define {{[^@]*}}@test_cxx_init_copy_list_plain_arr_idx
// CXX: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_cxx_init_copy_list_plain_arr_idx(struct Small arr[]) {
  struct Small a = {arr[0]};
}

// CXX-LABEL: define {{[^@]*}}@test_cxx_new_direct_plain_arr_idx
// CXX: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_cxx_new_direct_plain_arr_idx(struct Small arr[]) {
  struct Small *a = new struct Small(arr[0]);
  delete a;
}

// CXX-LABEL: define {{[^@]*}}@test_cxx_new_brace_plain_arr_idx
// CXX: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_cxx_new_brace_plain_arr_idx(struct Small arr[]) {
  struct Small *a = new struct Small{arr[0]};
  delete a;
}

// CXX-LABEL: define {{[^@]*}}@test_cxx_functional_cast_plain_arr_idx
// CXX: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_cxx_functional_cast_plain_arr_idx(struct Small arr[]) {
  (void)Small(arr[0]);
}

// CXX-LABEL: define {{[^@]*}}@test_cxx_functional_cast_brace_plain_arr_idx
// CXX: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_cxx_functional_cast_brace_plain_arr_idx(struct Small arr[]) {
  (void)Small{arr[0]};
}

// CXX-LABEL: define {{[^@]*}}@test_cxx_static_cast_plain_arr_idx
// CXX: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_cxx_static_cast_plain_arr_idx(struct Small arr[]) {
  (void)static_cast<struct Small>(arr[0]);
}

// CXX-LABEL: define {{[^@]*}}@test_cxx_c_cast_plain_arr_idx
// CXX: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_cxx_c_cast_plain_arr_idx(struct Small arr[]) {
  (void)(struct Small)(arr[0]);
}

// CXX-LABEL: define {{[^@]*}}@test_cxx_member_init_plain_arr_idx
// CXX: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_cxx_member_init_plain_arr_idx(struct Small arr[]) {
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
__attribute__((noinline)) void test_cxx_variadic_plain_arr_idx(struct Small arr[]) {
  variadic_func(0, arr[0]);
}

// CXX-LABEL: define {{[^@]*}}@test_cxx_variadic_plain_deref_ptr
__attribute__((noinline)) void test_cxx_variadic_plain_deref_ptr(struct Small *ap) {
  variadic_func(0, *ap);
}

// --- Explicit operator= calls ---

// CXX-LABEL: define {{[^@]*}}@test_cxx_operator_assign_lhs_arr_idx
// CXX: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_cxx_operator_assign_lhs_arr_idx(struct Small *dest, struct Small arr[]) {
  dest->operator=(arr[0]);
}

// CXX-LABEL: define {{[^@]*}}@test_cxx_operator_assign_lhs_deref_ptr
// CXX: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_cxx_operator_assign_lhs_deref_ptr(struct Small *dest, struct Small *ap) {
  dest->operator=(*ap);
}

// CXX-LABEL: define {{[^@]*}}@test_cxx_operator_assign_obj_arr_idx
// CXX: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_cxx_operator_assign_obj_arr_idx(struct Small arr[]) {
  struct Small a;
  a.operator=(arr[0]);
}

// CXX-LABEL: define {{[^@]*}}@test_cxx_operator_assign_obj_deref_ptr
// CXX: call void @llvm.memcpy.p0.p0.i64
__attribute__((noinline)) void test_cxx_operator_assign_obj_deref_ptr(struct Small *ap) {
  struct Small a;
  a.operator=(*ap);
}

} // extern "C"

// --- Virtual base initialization (cannot be extern "C") ---
// Use a simple struct to ensure memcpy is used for the copy

struct VirtualBaseSimple {
  int x;
};

struct DerivedVirtualArrSimple : virtual VirtualBaseSimple {
  __attribute__((noinline)) DerivedVirtualArrSimple(VirtualBaseSimple arr[]) : VirtualBaseSimple(arr[0]) {}
};

struct DerivedVirtualPtrSimple : virtual VirtualBaseSimple {
  __attribute__((noinline)) DerivedVirtualPtrSimple(VirtualBaseSimple *ap) : VirtualBaseSimple(*ap) {}
};

// Force instantiation of constructors
// The constructors are emitted inline after their first use, so we check them
// in the order they appear: ArrSimple constructor, then wrapper for Ptr, then PtrSimple constructor

extern "C" {

// CXX-LABEL: define {{[^@]*}}@test_cxx_virtual_base_init_arr_idx
__attribute__((noinline)) void test_cxx_virtual_base_init_arr_idx(VirtualBaseSimple arr[]) {
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
