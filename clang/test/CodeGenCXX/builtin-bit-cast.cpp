// RUN: %clang_cc1 -no-opaque-pointers -std=c++2a -S -emit-llvm -o - -disable-llvm-passes -triple x86_64-apple-macos10.14 %s | FileCheck %s

void test_scalar(int &oper) {
  // CHECK-LABEL: define{{.*}} void @_Z11test_scalarRi
  __builtin_bit_cast(float, oper);

  // CHECK: [[OPER:%.*]] = alloca i32*
  // CHECK: [[REF:%.*]] = load i32*, i32**
  // CHECK-NEXT: [[CASTED:%.*]] = bitcast i32* [[REF]] to float*
  // CHECK-NEXT: load float, float* [[CASTED]]
}

struct two_ints {
  int x;
  int y;
};

unsigned long test_aggregate_to_scalar(two_ints &ti) {
  // CHECK-LABEL: define{{.*}} i64 @_Z24test_aggregate_to_scalarR8two_ints
  return __builtin_bit_cast(unsigned long, ti);

  // CHECK: [[TI_ADDR:%.*]] = alloca %struct.two_ints*, align 8
  // CHECK: [[TI_LOAD:%.*]] = load %struct.two_ints*, %struct.two_ints** [[TI_ADDR]]
  // CHECK-NEXT: [[CASTED:%.*]] = bitcast %struct.two_ints* [[TI_LOAD]] to i64*
  // CHECK-NEXT: load i64, i64* [[CASTED]]
}

struct two_floats {
  float x;
  float y;
};

two_floats test_aggregate_record(two_ints& ti) {
  // CHECK-LABEL: define{{.*}} <2 x float> @_Z21test_aggregate_recordR8two_int
   return __builtin_bit_cast(two_floats, ti);

  // CHECK: [[RETVAL:%.*]] = alloca %struct.two_floats, align 4
  // CHECK: [[TI:%.*]] = alloca %struct.two_ints*, align 8

  // CHECK: [[LOAD_TI:%.*]] = load %struct.two_ints*, %struct.two_ints** [[TI]]
  // CHECK: [[MEMCPY_SRC:%.*]] = bitcast %struct.two_ints* [[LOAD_TI]] to i8*
  // CHECK: [[MEMCPY_DST:%.*]] = bitcast %struct.two_floats* [[RETVAL]] to i8*
  // CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 [[MEMCPY_DST]], i8* align 4 [[MEMCPY_SRC]]
}

two_floats test_aggregate_array(int (&ary)[2]) {
  // CHECK-LABEL: define{{.*}} <2 x float> @_Z20test_aggregate_arrayRA2_i
  return __builtin_bit_cast(two_floats, ary);

  // CHECK: [[RETVAL:%.*]] = alloca %struct.two_floats, align 4
  // CHECK: [[ARY:%.*]] = alloca [2 x i32]*, align 8

  // CHECK: [[LOAD_ARY:%.*]] = load [2 x i32]*, [2 x i32]** [[ARY]]
  // CHECK: [[MEMCPY_SRC:%.*]] = bitcast [2 x i32]* [[LOAD_ARY]] to i8*
  // CHECK: [[MEMCPY_DST:%.*]] = bitcast %struct.two_floats* [[RETVAL]] to i8*
  // CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 [[MEMCPY_DST]], i8* align 4 [[MEMCPY_SRC]]
}

two_ints test_scalar_to_aggregate(unsigned long ul) {
  // CHECK-LABEL: define{{.*}} i64 @_Z24test_scalar_to_aggregatem
  return __builtin_bit_cast(two_ints, ul);

  // CHECK: [[TI:%.*]] = alloca %struct.two_ints, align 4
  // CHECK: [[TITMP:%.*]] = bitcast %struct.two_ints* [[TI]] to i8*
  // CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 [[TITMP]]
}

unsigned long test_complex(_Complex unsigned &cu) {
  // CHECK-LABEL: define{{.*}} i64 @_Z12test_complexRCj
  return __builtin_bit_cast(unsigned long, cu);

  // CHECK: [[REF_ALLOCA:%.*]] = alloca { i32, i32 }*, align 8
  // CHECK-NEXT: store { i32, i32 }* {{.*}}, { i32, i32 }** [[REF_ALLOCA]]
  // CHECK-NEXT: [[REF:%.*]] = load { i32, i32 }*, { i32, i32 }** [[REF_ALLOCA]]
  // CHECK-NEXT: [[CASTED:%.*]] = bitcast { i32, i32 }* [[REF]] to i64*
  // CHECK-NEXT: load i64, i64* [[CASTED]], align 4
}

_Complex unsigned test_to_complex(unsigned long &ul) {
  // CHECK-LABEL: define{{.*}} i64 @_Z15test_to_complexRm

  return __builtin_bit_cast(_Complex unsigned, ul);

  // CHECK: [[REF:%.*]] = alloca i64*
  // CHECK: [[LOAD_REF:%.*]] = load i64*, i64** [[REF]]
  // CHECK: [[CASTED:%.*]] = bitcast i64* [[LOAD_REF]] to { i32, i32 }*
}

unsigned long test_array(int (&ary)[2]) {
  // CHECK-LABEL: define{{.*}} i64 @_Z10test_arrayRA2_i
  return __builtin_bit_cast(unsigned long, ary);

  // CHECK: [[REF_ALLOCA:%.*]] = alloca [2 x i32]*
  // CHECK: [[LOAD_REF:%.*]] = load [2 x i32]*, [2 x i32]** [[REF_ALLOCA]]
  // CHECK: [[CASTED:%.*]] = bitcast [2 x i32]* [[LOAD_REF]] to i64*
  // CHECK: load i64, i64* [[CASTED]], align 4
}

two_ints test_rvalue_aggregate() {
  // CHECK-LABEL: define{{.*}} i64 @_Z21test_rvalue_aggregate
  return __builtin_bit_cast(two_ints, 42ul);

  // CHECK: [[TI:%.*]] = alloca %struct.two_ints, align 4
  // CHECK: [[CASTED:%.*]] = bitcast %struct.two_ints* [[TI]] to i8*
  // CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 [[CASTED]]
}
