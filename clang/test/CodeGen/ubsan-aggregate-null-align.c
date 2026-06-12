// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -fsanitize=null,alignment,array-bounds -Wno-array-bounds -std=c11 -O0 %s -o - | FileCheck %s --check-prefixes=CHECK,C
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -fsanitize=null,alignment,array-bounds -Wno-array-bounds -std=c++17 -x c++ -O0 %s -o - | FileCheck %s --check-prefixes=CHECK,CXX
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -fsanitize=null,alignment,array-bounds -Wno-array-bounds -std=c11 -O0 -DUSE_UNION %s -o - | FileCheck %s --check-prefixes=CHECK,C
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -fsanitize=null,alignment,array-bounds -Wno-array-bounds -std=c++17 -x c++ -O0 -DUSE_UNION %s -o - | FileCheck %s --check-prefixes=CHECK,CXX

#ifdef USE_UNION
union Agg { int x; };
#define AGG union Agg
#else
struct Agg { int x; };
#define AGG struct Agg
#endif

#ifdef __cplusplus
extern "C" {
#endif

// LHS checks - C only
// Note: In C++, aggregate assignment goes through operator=
// which is a different code path (CGExprCXX.cpp).
// FIXME: LHS checks for C++ will be addressed in a follow-up PR

// C-LABEL: define {{.*}}@test_lhs_ptrcheck_deref(
// C: [[DEST:%.*]] = load ptr, ptr %dest.addr
// C-NEXT: [[CMP:%.*]] = icmp ne ptr [[DEST]], null, !nosanitize
// C-NEXT: [[INT:%.*]] = ptrtoint ptr [[DEST]] to i64, !nosanitize
// C-NEXT: [[AND:%.*]] = and i64 [[INT]], 3, !nosanitize
// C-NEXT: [[ALIGN:%.*]] = icmp eq i64 [[AND]], 0, !nosanitize
// C-NEXT: [[OK:%.*]] = and i1 [[CMP]], [[ALIGN]], !nosanitize
// C-NEXT: br i1 [[OK]], label %cont, label %handler.type_mismatch
// C: handler.type_mismatch:
// C-NEXT: call void @__ubsan_handle_type_mismatch_v1_abort
// C: call void @llvm.memcpy
void test_lhs_ptrcheck_deref(AGG *dest) {
  AGG local = {0};
  *dest = local;
}

// C-LABEL: define {{.*}}@test_lhs_ptrcheck_subscript(
// C: call void @__ubsan_handle_type_mismatch_v1_abort
void test_lhs_ptrcheck_subscript(AGG arr[4]) {
  AGG local = {0};
  arr[0] = local;
}

// RHS checks - both C and C++

// CHECK-LABEL: define {{.*}}@test_rhs_ptrcheck_deref(
// CHECK: [[SRC:%.*]] = load ptr, ptr %src.addr
// CHECK-NEXT: [[CMP:%.*]] = icmp ne ptr [[SRC]], null, !nosanitize
// CHECK-NEXT: [[INT:%.*]] = ptrtoint ptr [[SRC]] to i64, !nosanitize
// CHECK-NEXT: [[AND:%.*]] = and i64 [[INT]], 3, !nosanitize
// CHECK-NEXT: [[ALIGN:%.*]] = icmp eq i64 [[AND]], 0, !nosanitize
// CHECK-NEXT: [[OK:%.*]] = and i1 [[CMP]], [[ALIGN]], !nosanitize
// CHECK-NEXT: br i1 [[OK]], label %cont, label %handler.type_mismatch
// CHECK: handler.type_mismatch:
// CHECK-NEXT: call void @__ubsan_handle_type_mismatch_v1_abort
// CHECK: cont:
// CHECK-NEXT: call void @llvm.memcpy
void test_rhs_ptrcheck_deref(AGG *src) {
  AGG local;
  local = *src;
  (void)local;
}

// CHECK-LABEL: define {{.*}}@test_rhs_ptrcheck_subscript(
// CHECK: call void @__ubsan_handle_type_mismatch_v1_abort
void test_rhs_ptrcheck_subscript(AGG arr[4]) {
  AGG local;
  local = arr[0];
  (void)local;
}

// RHS cases - handler call only

// CHECK-LABEL: define {{.*}}@test_init_from_ptr(
// CHECK: call void @__ubsan_handle_type_mismatch_v1_abort
void test_init_from_ptr(AGG *src) {
  AGG local = *src;
  (void)local;
}

// CHECK-LABEL: define {{.*}}@test_init_from_array(
// CHECK: call void @__ubsan_handle_type_mismatch_v1_abort
void test_init_from_array(AGG arr[4]) {
  AGG local = arr[0];
  (void)local;
}

// Array bounds - out-of-bounds access (RHS)

// CHECK-LABEL: define {{.*}}@test_oob_rhs(
// C: br i1 false, label %cont, label %handler.out_of_bounds
// CXX: br i1 true, label %cont, label %handler.out_of_bounds
// CHECK: handler.out_of_bounds:
// CHECK-NEXT: call void @__ubsan_handle_out_of_bounds_abort
// CHECK: handler.type_mismatch:
// CHECK-NEXT: call void @__ubsan_handle_type_mismatch_v1_abort
// CHECK: call void @llvm.memcpy
void test_oob_rhs(void) {
  AGG arr[4];
  AGG local;
  local = arr[4];
  (void)local;
}

// Array bounds - out-of-bounds access (LHS)
// FIXME: LHS checks for C++ will be addressed in a follow-up PR.

// CHECK-LABEL: define {{.*}}@test_oob_lhs(
// C: br i1 false, label %cont, label %handler.out_of_bounds
// CXX: br i1 true, label %cont, label %handler.out_of_bounds
// CHECK: handler.out_of_bounds:
// CHECK-NEXT: call void @__ubsan_handle_out_of_bounds_abort
// C: handler.type_mismatch:
// C-NEXT: call void @__ubsan_handle_type_mismatch_v1_abort
// CHECK: call void @llvm.memcpy
void test_oob_lhs(void) {
  AGG arr[4];
  AGG local = {0};
  arr[4] = local;
  (void)arr;
}

#ifdef __cplusplus
}
#endif

// C++ RHS cases - handler call only

#ifdef __cplusplus

extern "C" {

// CXX-LABEL: define {{.*}}@test_cxx_direct_init(
// CXX: call void @__ubsan_handle_type_mismatch_v1_abort
void test_cxx_direct_init(AGG *src) {
  AGG local(*src);
  (void)local;
}

// CXX-LABEL: define {{.*}}@test_cxx_brace_init(
// CXX: call void @__ubsan_handle_type_mismatch_v1_abort
void test_cxx_brace_init(AGG *src) {
  AGG local{*src};
  (void)local;
}

// CXX-LABEL: define {{.*}}@test_cxx_new(
// operator new return value check (existing)
// CXX: call void @__ubsan_handle_type_mismatch_v1_abort
// Instrumentation for *src access
// CXX: [[SRC:%.*]] = load ptr, ptr %src.addr
// CXX-NEXT: [[CMP:%.*]] = icmp ne ptr [[SRC]], null, !nosanitize
// CXX-NEXT: [[INT:%.*]] = ptrtoint ptr [[SRC]] to i64, !nosanitize
// CXX-NEXT: [[AND:%.*]] = and i64 [[INT]], 3, !nosanitize
// CXX-NEXT: [[ALIGN:%.*]] = icmp eq i64 [[AND]], 0, !nosanitize
// CXX-NEXT: [[OK:%.*]] = and i1 [[CMP]], [[ALIGN]], !nosanitize
// CXX-NEXT: br i1 [[OK]], label %{{.*}}, label %[[MISMATCH:[^,]*]]
// CXX: [[MISMATCH]]:
// CXX-NEXT: call void @__ubsan_handle_type_mismatch_v1_abort
// CXX: call void @llvm.memcpy
void test_cxx_new(AGG *src) {
  AGG *p = new AGG(*src);
  delete p;
}

}

#endif
