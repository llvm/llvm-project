// RUN: %clang_cc1 -std=c++2a -emit-llvm -o - -disable-llvm-passes -triple x86_64-apple-macos10.14 %s | FileCheck %s

void test_scalar(int &oper) {
  // CHECK-LABEL: define{{.*}} void @_Z11test_scalarRi
  __builtin_bit_cast(float, oper);

  // CHECK: [[OPER:%.*]] = alloca ptr
  // CHECK: [[REF:%.*]] = load ptr, ptr
  // CHECK-NEXT: load float, ptr [[REF]]
}

struct two_ints {
  int x;
  int y;
};

unsigned long test_aggregate_to_scalar(two_ints &ti) {
  // CHECK-LABEL: define{{.*}} i64 @_Z24test_aggregate_to_scalarR8two_ints
  return __builtin_bit_cast(unsigned long, ti);

  // CHECK: [[TI_ADDR:%.*]] = alloca ptr, align 8
  // CHECK: [[TI_LOAD:%.*]] = load ptr, ptr [[TI_ADDR]]
  // CHECK-NEXT: load i64, ptr [[TI_LOAD]]
}

struct two_floats {
  float x;
  float y;
};

two_floats test_aggregate_record(two_ints& ti) {
  // CHECK-LABEL: define{{.*}} <2 x float> @_Z21test_aggregate_recordR8two_int
   return __builtin_bit_cast(two_floats, ti);

  // CHECK: [[RETVAL:%.*]] = alloca %struct.two_floats, align 4
  // CHECK: [[TI:%.*]] = alloca ptr, align 8

  // CHECK: [[LOAD_TI:%.*]] = load ptr, ptr [[TI]]
  // CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 4 [[RETVAL]], ptr align 4 [[LOAD_TI]]
}

two_floats test_aggregate_array(int (&ary)[2]) {
  // CHECK-LABEL: define{{.*}} <2 x float> @_Z20test_aggregate_arrayRA2_i
  return __builtin_bit_cast(two_floats, ary);

  // CHECK: [[RETVAL:%.*]] = alloca %struct.two_floats, align 4
  // CHECK: [[ARY:%.*]] = alloca ptr, align 8

  // CHECK: [[LOAD_ARY:%.*]] = load ptr, ptr [[ARY]]
  // CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 4 [[RETVAL]], ptr align 4 [[LOAD_ARY]]
}

two_ints test_scalar_to_aggregate(unsigned long ul) {
  // CHECK-LABEL: define{{.*}} i64 @_Z24test_scalar_to_aggregatem
  return __builtin_bit_cast(two_ints, ul);

  // CHECK: [[TI:%.*]] = alloca %struct.two_ints, align 4
  // CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 4 [[TI]]
}

unsigned long test_complex(_Complex unsigned &cu) {
  // CHECK-LABEL: define{{.*}} i64 @_Z12test_complexRCj
  return __builtin_bit_cast(unsigned long, cu);

  // CHECK: [[REF_ALLOCA:%.*]] = alloca ptr, align 8
  // CHECK-NEXT: store ptr {{.*}}, ptr [[REF_ALLOCA]]
  // CHECK-NEXT: [[REF:%.*]] = load ptr, ptr [[REF_ALLOCA]]
  // CHECK-NEXT: load i64, ptr [[REF]], align 4
}

_Complex unsigned test_to_complex(unsigned long &ul) {
  // CHECK-LABEL: define{{.*}} i64 @_Z15test_to_complexRm

  return __builtin_bit_cast(_Complex unsigned, ul);

  // CHECK: [[REF:%.*]] = alloca ptr
  // CHECK: [[LOAD_REF:%.*]] = load ptr, ptr [[REF]]
}

unsigned long test_array(int (&ary)[2]) {
  // CHECK-LABEL: define{{.*}} i64 @_Z10test_arrayRA2_i
  return __builtin_bit_cast(unsigned long, ary);

  // CHECK: [[REF_ALLOCA:%.*]] = alloca ptr
  // CHECK: [[LOAD_REF:%.*]] = load ptr, ptr [[REF_ALLOCA]]
  // CHECK: load i64, ptr [[LOAD_REF]], align 4
}

two_ints test_rvalue_aggregate() {
  // CHECK-LABEL: define{{.*}} i64 @_Z21test_rvalue_aggregate
  return __builtin_bit_cast(two_ints, 42ul);

  // CHECK: [[TI:%.*]] = alloca %struct.two_ints, align 4
  // CHECK: call void @llvm.memcpy.p0.p0.i64(ptr align 4 [[TI]]
}
