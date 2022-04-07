// RUN: %clang_cc1 -no-opaque-pointers -x c -triple x86_64-apple-darwin10 -w -emit-llvm -o - %s -fsanitize=pointer-overflow | FileCheck %s --check-prefixes=CHECK,CHECK-C
// RUN: %clang_cc1 -no-opaque-pointers -x c++ -triple x86_64-apple-darwin10 -w -emit-llvm -o - %s -fsanitize=pointer-overflow | FileCheck %s --check-prefixes=CHECK,CHECK-CPP

#ifdef __cplusplus
extern "C" {
#endif

// CHECK-LABEL: define{{.*}} void @fixed_len_array
void fixed_len_array(int k) {
  // CHECK: getelementptr inbounds [10 x [10 x i32]], [10 x [10 x i32]]* [[ARR:%.*]], i64 0, i64 [[IDXPROM:%.*]]
  // CHECK-NEXT: [[SMUL:%.*]] = call { i64, i1 } @llvm.smul.with.overflow.i64(i64 40, i64 [[IDXPROM]]), !nosanitize
  // CHECK-NEXT: [[SMULOFLOW:%.*]] = extractvalue { i64, i1 } [[SMUL]], 1, !nosanitize
  // CHECK-NEXT: [[OR:%.+]] = or i1 [[SMULOFLOW]], false, !nosanitize
  // CHECK-NEXT: [[SMULVAL:%.*]] = extractvalue { i64, i1 } [[SMUL]], 0, !nosanitize
  // CHECK-NEXT: [[BASE:%.*]] = ptrtoint [10 x [10 x i32]]* [[ARR]] to i64, !nosanitize
  // CHECK-NEXT: [[COMPGEP:%.*]] = add i64 [[BASE]], [[SMULVAL]], !nosanitize
  // CHECK: call void @__ubsan_handle_pointer_overflow{{.*}}, i64 [[BASE]], i64 [[COMPGEP]]){{.*}}, !nosanitize

  // CHECK: getelementptr inbounds [10 x i32], [10 x i32]* {{.*}}, i64 0, i64 [[IDXPROM1:%.*]]
  // CHECK-NEXT: @llvm.smul.with.overflow.i64(i64 4, i64 [[IDXPROM1]]), !nosanitize
  // CHECK: call void @__ubsan_handle_pointer_overflow{{.*}}

  int arr[10][10];
  arr[k][k];
}

// CHECK-LABEL: define{{.*}} void @variable_len_array
void variable_len_array(int n, int k) {
  // CHECK: getelementptr inbounds i32, i32* {{.*}}, i64 [[IDXPROM:%.*]]
  // CHECK-NEXT: @llvm.smul.with.overflow.i64(i64 4, i64 [[IDXPROM]]), !nosanitize
  // CHECK: call void @__ubsan_handle_pointer_overflow{{.*}}

  // CHECK: getelementptr inbounds i32, i32* {{.*}}, i64 [[IDXPROM1:%.*]]
  // CHECK-NEXT: @llvm.smul.with.overflow.i64(i64 4, i64 [[IDXPROM1]]), !nosanitize
  // CHECK: call void @__ubsan_handle_pointer_overflow{{.*}}

  int arr[n][n];
  arr[k][k];
}

// CHECK-LABEL: define{{.*}} void @pointer_array
void pointer_array(int **arr, int k) {
  // CHECK: @llvm.smul.with.overflow.i64(i64 8, i64 {{.*}}), !nosanitize
  // CHECK: call void @__ubsan_handle_pointer_overflow{{.*}}

  // CHECK: @llvm.smul.with.overflow.i64(i64 4, i64 {{.*}}), !nosanitize
  // CHECK: call void @__ubsan_handle_pointer_overflow{{.*}}

  arr[k][k];
}

// CHECK-LABEL: define{{.*}} void @pointer_array_unsigned_indices
void pointer_array_unsigned_indices(int **arr, unsigned k) {
  // CHECK: icmp uge
  // CHECK-NOT: select
  // CHECK: call void @__ubsan_handle_pointer_overflow{{.*}}
  // CHECK: icmp uge
  // CHECK-NOT: select
  // CHECK: call void @__ubsan_handle_pointer_overflow{{.*}}
  arr[k][k];
}

// CHECK-LABEL: define{{.*}} void @pointer_array_mixed_indices
void pointer_array_mixed_indices(int **arr, int i, unsigned j) {
  // CHECK: select
  // CHECK: call void @__ubsan_handle_pointer_overflow{{.*}}
  // CHECK-NOT: select
  // CHECK: call void @__ubsan_handle_pointer_overflow{{.*}}
  arr[i][j];
}

struct S1 {
  int pad1;
  union {
    char leaf;
    struct S1 *link;
  } u;
  struct S1 *arr;
};

// TODO: Currently, structure GEPs are not checked, so there are several
// potentially unsafe GEPs here which we don't instrument.
//
// CHECK-LABEL: define{{.*}} void @struct_index
void struct_index(struct S1 *p) {
  // CHECK: getelementptr inbounds %struct.S1, %struct.S1* [[P:%.*]], i64 10
  // CHECK-NEXT: [[BASE:%.*]] = ptrtoint %struct.S1* [[P]] to i64, !nosanitize
  // CHECK-NEXT: [[COMPGEP:%.*]] = add i64 [[BASE]], 240, !nosanitize
  // CHECK: select
  // CHECK: @__ubsan_handle_pointer_overflow{{.*}} i64 [[BASE]], i64 [[COMPGEP]]) {{.*}}, !nosanitize

  // CHECK-NOT: @__ubsan_handle_pointer_overflow

  p->arr[10].u.link->u.leaf;
}

typedef void (*funcptr_t)(void);

// CHECK-LABEL: define{{.*}} void @function_pointer_arith
void function_pointer_arith(funcptr_t *p, int k) {
  // CHECK: add i64 {{.*}}, 8, !nosanitize
  // CHECK-NOT: select
  // CHECK: @__ubsan_handle_pointer_overflow{{.*}}
  ++p;

  // CHECK: @llvm.smul.with.overflow.i64(i64 8, i64 {{.*}}), !nosanitize
  // CHECK: select
  // CHECK: call void @__ubsan_handle_pointer_overflow{{.*}}
  p + k;
}

// CHECK-LABEL: define{{.*}} void @dont_emit_checks_for_no_op_GEPs
// CHECK-C: __ubsan_handle_pointer_overflow
// CHECK-CPP-NOT: __ubsan_handle_pointer_overflow
void dont_emit_checks_for_no_op_GEPs(char *p) {
  &p[0];

  int arr[10][10];
  &arr[0][0];
}

#ifdef __cplusplus
}
#endif
