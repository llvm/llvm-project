// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -w -emit-llvm -o - %s -fsanitize=pointer-overflow | FileCheck %s

// CHECK-LABEL: define void @unary_arith
void unary_arith(char *p) {
  // CHECK:  [[BASE:%.*]] = ptrtoint i8* {{.*}} to i64, !nosanitize
  // CHECK-NEXT: [[COMPGEP:%.*]] = add i64 [[BASE]], 1, !nosanitize
  // CHECK-NEXT: [[POSVALID:%.*]] = icmp uge i64 [[COMPGEP]], [[BASE]], !nosanitize
  // CHECK-NEXT: br i1 [[POSVALID]]{{.*}}, !nosanitize
  // CHECK: call void @__ubsan_handle_pointer_overflow{{.*}}, i64 [[BASE]], i64 [[COMPGEP]]){{.*}}, !nosanitize
  ++p;

  // CHECK: ptrtoint i8* {{.*}} to i64, !nosanitize
  // CHECK-NEXT: add i64 {{.*}}, -1, !nosanitize
  // CHECK: select i1 false{{.*}}, !nosanitize
  // CHECK: call void @__ubsan_handle_pointer_overflow{{.*}}
  --p;

  // CHECK-NOT: select
  // CHECK: call void @__ubsan_handle_pointer_overflow{{.*}}
  p++;

  // CHECK: select
  // CHECK: call void @__ubsan_handle_pointer_overflow{{.*}}
  p--;
}

// CHECK-LABEL: define void @binary_arith
void binary_arith(char *p, int i) {
  // CHECK: [[SMUL:%.*]] = call { i64, i1 } @llvm.smul.with.overflow.i64(i64 1, i64 %{{.*}}), !nosanitize
  // CHECK-NEXT: [[SMULOFLOW:%.*]] = extractvalue { i64, i1 } [[SMUL]], 1, !nosanitize
  // CHECK-NEXT: [[SMULVAL:%.*]] = extractvalue { i64, i1 } [[SMUL]], 0, !nosanitize
  // CHECK-NEXT: [[BASE:%.*]] = ptrtoint i8* {{.*}} to i64, !nosanitize
  // CHECK-NEXT: [[COMPGEP:%.*]] = add i64 [[BASE]], [[SMULVAL]], !nosanitize
  // CHECK-NEXT: [[OFFSETVALID:%.*]] = xor i1 [[SMULOFLOW]], true, !nosanitize
  // CHECK-NEXT: [[POSVALID:%.*]] = icmp uge i64 [[COMPGEP]], [[BASE]], !nosanitize
  // CHECK-NEXT: [[POSOFFSET:%.*]] = icmp sge i64 [[SMULVAL]], 0, !nosanitize
  // CHECK-NEXT: [[NEGVALID:%.*]] = icmp ult i64 [[COMPGEP]], [[BASE]], !nosanitize
  // CHECK-NEXT: [[DIFFVALID:%.*]] = select i1 [[POSOFFSET]], i1 [[POSVALID]], i1 [[NEGVALID]], !nosanitize
  // CHECK: [[VALID:%.*]] = and i1 [[DIFFVALID]], [[OFFSETVALID]], !nosanitize
  // CHECK-NEXT: br i1 [[VALID]]{{.*}}, !nosanitize
  // CHECK: call void @__ubsan_handle_pointer_overflow{{.*}}, i64 [[BASE]], i64 [[COMPGEP]]){{.*}}, !nosanitize
  p + i;

  // CHECK: [[OFFSET:%.*]] = sub i64 0, {{.*}}
  // CHECK-NEXT: getelementptr inbounds {{.*}} [[OFFSET]]
  // CHECK: select
  // CHECK: call void @__ubsan_handle_pointer_overflow{{.*}}
  p - i;
}

// CHECK-LABEL: define void @binary_arith_unsigned
void binary_arith_unsigned(char *p, unsigned i) {
  // CHECK: [[SMUL:%.*]] = call { i64, i1 } @llvm.smul.with.overflow.i64(i64 1, i64 %{{.*}}), !nosanitize
  // CHECK-NEXT: [[SMULOFLOW:%.*]] = extractvalue { i64, i1 } [[SMUL]], 1, !nosanitize
  // CHECK-NEXT: [[SMULVAL:%.*]] = extractvalue { i64, i1 } [[SMUL]], 0, !nosanitize
  // CHECK-NEXT: [[BASE:%.*]] = ptrtoint i8* {{.*}} to i64, !nosanitize
  // CHECK-NEXT: [[COMPGEP:%.*]] = add i64 [[BASE]], [[SMULVAL]], !nosanitize
  // CHECK-NEXT: [[OFFSETVALID:%.*]] = xor i1 [[SMULOFLOW]], true, !nosanitize
  // CHECK-NEXT: [[POSVALID:%.*]] = icmp uge i64 [[COMPGEP]], [[BASE]], !nosanitize
  // CHECK: [[VALID:%.*]] = and i1 [[POSVALID]], [[OFFSETVALID]], !nosanitize
  // CHECK-NEXT: br i1 [[VALID]]{{.*}}, !nosanitize
  // CHECK: call void @__ubsan_handle_pointer_overflow{{.*}}, i64 [[BASE]], i64 [[COMPGEP]]){{.*}}, !nosanitize
  p + i;

  // CHECK: [[OFFSET:%.*]] = sub i64 0, {{.*}}
  // CHECK-NEXT: getelementptr inbounds {{.*}} [[OFFSET]]
  // CHECK: select
  // CHECK: call void @__ubsan_handle_pointer_overflow{{.*}}
  p - i;
}

// CHECK-LABEL: define void @fixed_len_array
void fixed_len_array(int k) {
  // CHECK: getelementptr inbounds [10 x [10 x i32]], [10 x [10 x i32]]* [[ARR:%.*]], i64 0, i64 [[IDXPROM:%.*]]
  // CHECK-NEXT: [[SMUL:%.*]] = call { i64, i1 } @llvm.smul.with.overflow.i64(i64 40, i64 [[IDXPROM]]), !nosanitize
  // CHECK-NEXT: [[SMULOFLOW:%.*]] = extractvalue { i64, i1 } [[SMUL]], 1, !nosanitize
  // CHECK-NEXT: [[SMULVAL:%.*]] = extractvalue { i64, i1 } [[SMUL]], 0, !nosanitize
  // CHECK-NEXT: [[BASE:%.*]] = ptrtoint [10 x [10 x i32]]* [[ARR]] to i64, !nosanitize
  // CHECK-NEXT: [[COMPGEP:%.*]] = add i64 [[BASE]], [[SMULVAL]], !nosanitize
  // CHECK-NEXT: [[OFFSETVALID:%.*]] = xor i1 [[SMULOFLOW]], true, !nosanitize
  // CHECK-NEXT: [[POSVALID:%.*]] = icmp uge i64 [[COMPGEP]], [[BASE]], !nosanitize
  // CHECK-NEXT: [[POSOFFSET:%.*]] = icmp sge i64 [[SMULVAL]], 0, !nosanitize
  // CHECK-NEXT: [[NEGVALID:%.*]] = icmp ult i64 [[COMPGEP]], [[BASE]], !nosanitize
  // CHECK-NEXT: [[DIFFVALID:%.*]] = select i1 [[POSOFFSET]], i1 [[POSVALID]], i1 [[NEGVALID]], !nosanitize
  // CHECK: [[VALID:%.*]] = and i1 [[DIFFVALID]], [[OFFSETVALID]], !nosanitize
  // CHECK-NEXT: br i1 [[VALID]]{{.*}}, !nosanitize
  // CHECK: call void @__ubsan_handle_pointer_overflow{{.*}}, i64 [[BASE]], i64 [[COMPGEP]]){{.*}}, !nosanitize

  // CHECK: getelementptr inbounds [10 x i32], [10 x i32]* {{.*}}, i64 0, i64 [[IDXPROM1:%.*]]
  // CHECK-NEXT: @llvm.smul.with.overflow.i64(i64 4, i64 [[IDXPROM1]]), !nosanitize
  // CHECK: call void @__ubsan_handle_pointer_overflow{{.*}}

  int arr[10][10];
  arr[k][k];
}

// CHECK-LABEL: define void @variable_len_array
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

// CHECK-LABEL: define void @pointer_array
void pointer_array(int **arr, int k) {
  // CHECK: @llvm.smul.with.overflow.i64(i64 8, i64 {{.*}}), !nosanitize
  // CHECK: call void @__ubsan_handle_pointer_overflow{{.*}}
 
  // CHECK: @llvm.smul.with.overflow.i64(i64 4, i64 {{.*}}), !nosanitize
  // CHECK: call void @__ubsan_handle_pointer_overflow{{.*}}

  arr[k][k];
}

// CHECK-LABEL: define void @pointer_array_unsigned_indices
void pointer_array_unsigned_indices(int **arr, unsigned k) {
  // CHECK-NOT: select
  // CHECK: call void @__ubsan_handle_pointer_overflow{{.*}}
  // CHECK-NOT: select
  // CHECK: call void @__ubsan_handle_pointer_overflow{{.*}}
  arr[k][k];
}

// CHECK-LABEL: define void @pointer_array_mixed_indices
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
// CHECK-LABEL: define void @struct_index
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

// CHECK-LABEL: define void @function_pointer_arith
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

// CHECK-LABEL: define void @variable_len_array_arith
void variable_len_array_arith(int n, int k) {
  int vla[n];
  int (*p)[n] = &vla;

  // CHECK: getelementptr inbounds i32, i32* {{.*}}, i64 [[INC:%.*]]
  // CHECK: @llvm.smul.with.overflow.i64(i64 4, i64 [[INC]]), !nosanitize
  // CHECK-NOT: select
  // CHECK: call void @__ubsan_handle_pointer_overflow{{.*}}
  ++p;

  // CHECK: getelementptr inbounds i32, i32* {{.*}}, i64 [[IDXPROM:%.*]]
  // CHECK: @llvm.smul.with.overflow.i64(i64 4, i64 [[IDXPROM]]), !nosanitize
  // CHECK: select
  // CHECK: call void @__ubsan_handle_pointer_overflow{{.*}}
  p + k;
}

// CHECK-LABEL: define void @objc_id
void objc_id(id *p) {
  // CHECK: add i64 {{.*}}, 8, !nosanitize
  // CHECK-NOT: select
  // CHECK: @__ubsan_handle_pointer_overflow{{.*}}
  p++;
}

// CHECK-LABEL: define void @dont_emit_checks_for_no_op_GEPs
// CHECK-NOT: __ubsan_handle_pointer_overflow
void dont_emit_checks_for_no_op_GEPs(char *p) {
  &p[0];

  int arr[10][10];
  &arr[0][0];
}
