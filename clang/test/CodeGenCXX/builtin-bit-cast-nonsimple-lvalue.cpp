// RUN: %clang_cc1 -std=c++20 -fenable-matrix -emit-llvm -disable-llvm-passes -triple x86_64-unknown-linux-gnu %s -o - | FileCheck %s

// An operand without an address of its own is loaded through the lvalue and
// materialized, rather than read from the start of the containing object.

typedef float v4f __attribute__((ext_vector_type(4)));
typedef float gv4f __attribute__((vector_size(16)));
typedef float m2x2f __attribute__((matrix_type(2, 2)));

// CHECK-LABEL: define{{.*}} i32 @_Z18ext_vector_elementDv4_f(
// CHECK:         [[TMP:%.*]] = alloca float
// CHECK:         [[VEC:%.*]] = load <4 x float>, ptr %v.addr
// CHECK-NEXT:    [[ELT:%.*]] = extractelement <4 x float> [[VEC]], i64 2
// CHECK-NEXT:    store float [[ELT]], ptr [[TMP]]
// CHECK-NEXT:    [[RES:%.*]] = load i32, ptr [[TMP]]
// CHECK-NEXT:    ret i32 [[RES]]
unsigned ext_vector_element(v4f v) {
  return __builtin_bit_cast(unsigned, v.z);
}

// CHECK-LABEL: define{{.*}} i32 @_Z20ext_vector_subscriptDv4_f(
// CHECK:         [[TMP:%.*]] = alloca float
// CHECK:         [[VEC:%.*]] = load <4 x float>, ptr %v.addr
// CHECK-NEXT:    [[ELT:%.*]] = extractelement <4 x float> [[VEC]], i32 1
// CHECK-NEXT:    store float [[ELT]], ptr [[TMP]]
// CHECK-NEXT:    [[RES:%.*]] = load i32, ptr [[TMP]]
// CHECK-NEXT:    ret i32 [[RES]]
unsigned ext_vector_subscript(v4f v) {
  return __builtin_bit_cast(unsigned, v[1]);
}

// CHECK-LABEL: define{{.*}} i64 @_Z18ext_vector_swizzleDv4_f(
// CHECK:         [[TMP:%.*]] = alloca <2 x float>
// CHECK:         [[VEC:%.*]] = load <4 x float>, ptr %v.addr
// CHECK-NEXT:    [[ELTS:%.*]] = shufflevector <4 x float> [[VEC]], <4 x float> poison, <2 x i32> <i32 2, i32 3>
// CHECK-NEXT:    store <2 x float> [[ELTS]], ptr [[TMP]]
// CHECK-NEXT:    [[RES:%.*]] = load i64, ptr [[TMP]]
// CHECK-NEXT:    ret i64 [[RES]]
unsigned long long ext_vector_swizzle(v4f v) {
  return __builtin_bit_cast(unsigned long long, v.zw);
}

// CHECK-LABEL: define{{.*}} i32 @_Z20gnu_vector_subscriptDv4_fi(
// CHECK:         [[TMP:%.*]] = alloca float
// CHECK:         [[VEC:%.*]] = load <4 x float>, ptr %v.addr
// CHECK-NEXT:    [[IDX:%.*]] = load i32, ptr %i.addr
// CHECK-NEXT:    [[ELT:%.*]] = extractelement <4 x float> [[VEC]], i32 [[IDX]]
// CHECK-NEXT:    store float [[ELT]], ptr [[TMP]]
// CHECK-NEXT:    [[RES:%.*]] = load i32, ptr [[TMP]]
// CHECK-NEXT:    ret i32 [[RES]]
unsigned gnu_vector_subscript(gv4f v, int i) {
  return __builtin_bit_cast(unsigned, v[i]);
}

// CHECK-LABEL: define{{.*}} i32 @_Z14matrix_elementu11matrix_typeILm2ELm2EfE(
// CHECK:         [[TMP:%.*]] = alloca float
// CHECK:         [[MAT:%.*]] = load <4 x float>, ptr %m.addr
// CHECK-NEXT:    [[ELT:%.*]] = extractelement <4 x float> [[MAT]], i64 3
// CHECK-NEXT:    store float [[ELT]], ptr [[TMP]]
// CHECK-NEXT:    [[RES:%.*]] = load i32, ptr [[TMP]]
// CHECK-NEXT:    ret i32 [[RES]]
unsigned matrix_element(m2x2f m) {
  return __builtin_bit_cast(unsigned, m[1][1]);
}

struct S {
  unsigned a : 4, b : 12;
};

// CHECK-LABEL: define{{.*}} i32 @_Z9bit_field1S(
// CHECK:         [[TMP:%.*]] = alloca i32
// CHECK:         [[LOAD:%.*]] = load i16, ptr %s
// CHECK-NEXT:    [[SHR:%.*]] = lshr i16 [[LOAD]], 4
// CHECK-NEXT:    [[EXT:%.*]] = zext i16 [[SHR]] to i32
// CHECK-NEXT:    store i32 [[EXT]], ptr [[TMP]]
// CHECK-NEXT:    [[RES:%.*]] = load i32, ptr [[TMP]]
// CHECK-NEXT:    ret i32 [[RES]]
unsigned bit_field(S s) {
  return __builtin_bit_cast(unsigned, s.b);
}

register unsigned long stack_pointer asm("rsp");

// CHECK-LABEL: define{{.*}} i64 @_Z15global_registerv(
// CHECK:         [[TMP:%.*]] = alloca i64
// CHECK:         [[REG:%.*]] = call i64 @llvm.read_register.i64(
// CHECK-NEXT:    store i64 [[REG]], ptr [[TMP]]
// CHECK-NEXT:    [[RES:%.*]] = load i64, ptr [[TMP]]
// CHECK-NEXT:    ret i64 [[RES]]
long global_register() {
  return __builtin_bit_cast(long, stack_pointer);
}
