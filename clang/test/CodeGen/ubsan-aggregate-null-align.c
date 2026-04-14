// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -fsanitize=null,alignment,array-bounds -std=c11 -O0 %s -o %t.c.ll && FileCheck %s --check-prefixes=CHECK,C < %t.c.ll
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -fsanitize=null,alignment,array-bounds -std=c++17 -x c++ -O0 %s -o %t.cxx.ll && FileCheck %s --check-prefixes=CHECK,CXX < %t.cxx.ll
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -fsanitize=null,alignment,array-bounds -std=c11 -O0 -DUSE_UNION %s -o %t.union.ll && FileCheck %s --check-prefixes=CHECK,C < %t.union.ll

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

// C-LABEL: define {{.*}}@test_lhs_ptr(
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
void test_lhs_ptr(AGG *dest) {
  AGG local = {0};
  *dest = local;
}

// C-LABEL: define {{.*}}@test_lhs_array(
// C: [[ARR:%.*]] = load ptr, ptr %arr.addr
// C-NEXT: [[DEST:%.*]] = getelementptr inbounds %{{(struct|union)}}.Agg, ptr [[ARR]], i64 0
// C-NEXT: [[CMP:%.*]] = icmp ne ptr [[DEST]], null, !nosanitize
// C: handler.type_mismatch:
// C-NEXT: call void @__ubsan_handle_type_mismatch_v1_abort
// C: call void @llvm.memcpy
void test_lhs_array(AGG arr[4]) {
  AGG local = {0};
  arr[0] = local;
}

// RHS checks - both C and C++

// CHECK-LABEL: define {{.*}}@test_rhs_ptr(
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
void test_rhs_ptr(AGG *src) {
  AGG local;
  local = *src;
  (void)local;
}

// CHECK-LABEL: define {{.*}}@test_rhs_array(
// CHECK: [[ARR:%.*]] = load ptr, ptr %arr.addr
// CHECK-NEXT: [[SRC:%.*]] = getelementptr inbounds %{{(struct|union)}}.Agg, ptr [[ARR]], i64 0
// CHECK-NEXT: [[CMP:%.*]] = icmp ne ptr [[SRC]], null, !nosanitize
// CHECK: handler.type_mismatch:
// CHECK-NEXT: call void @__ubsan_handle_type_mismatch_v1_abort
// CHECK: call void @llvm.memcpy
void test_rhs_array(AGG arr[4]) {
  AGG local;
  local = arr[0];
  (void)local;
}

// RHS cases - handler call only

// CHECK-LABEL: define {{.*}}@test_init_from_ptr(
// CHECK: handler.type_mismatch:
// CHECK-NEXT: call void @__ubsan_handle_type_mismatch_v1_abort
// CHECK: call void @llvm.memcpy
void test_init_from_ptr(AGG *src) {
  AGG local = *src;
  (void)local;
}

// CHECK-LABEL: define {{.*}}@test_init_from_array(
// CHECK: handler.type_mismatch:
// CHECK-NEXT: call void @__ubsan_handle_type_mismatch_v1_abort
// CHECK: call void @llvm.memcpy
void test_init_from_array(AGG arr[4]) {
  AGG local = arr[0];
  (void)local;
}

// Array bounds - out-of-bounds access

// CHECK-LABEL: define {{.*}}@test_oob_rhs(
// CHECK: br i1 {{(true|false)}}, label %cont, label %handler.out_of_bounds
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

#ifdef __cplusplus
}
#endif


// C++ - handler call only

#ifdef __cplusplus

extern "C" {

// CXX-LABEL: define {{.*}}@test_cxx_direct_init(
// CXX: handler.type_mismatch:
// CXX-NEXT: call void @__ubsan_handle_type_mismatch_v1_abort
// CXX: call void @llvm.memcpy
void test_cxx_direct_init(AGG *src) {
  AGG local(*src);
  (void)local;
}

// CXX-LABEL: define {{.*}}@test_cxx_brace_init(
// CXX: handler.type_mismatch:
// CXX-NEXT: call void @__ubsan_handle_type_mismatch_v1_abort
// CXX: call void @llvm.memcpy
void test_cxx_brace_init(AGG *src) {
  AGG local{*src};
  (void)local;
}

// CXX-LABEL: define {{.*}}@test_cxx_new(
// CXX: handler.type_mismatch:
// CXX-NEXT: call void @__ubsan_handle_type_mismatch_v1_abort
// CXX: call void @llvm.memcpy
void test_cxx_new(AGG *src) {
  AGG *p = new AGG(*src);
  delete p;
}

}

#endif
