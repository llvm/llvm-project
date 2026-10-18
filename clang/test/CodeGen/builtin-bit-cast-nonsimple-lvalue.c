// RUN: %clang_cc1 -O -emit-llvm -disable-llvm-passes -triple x86_64-unknown-linux-gnu %s -o - | FileCheck %s

typedef float v4f __attribute__((ext_vector_type(4)));

// CHECK-LABEL: define{{.*}} i32 @ext_vector_element(
// CHECK:         [[TMP:%ref.tmp.*]] = alloca float
// CHECK:         call void @llvm.lifetime.start.p0(ptr [[TMP]])
// CHECK-NEXT:    [[VEC:%.*]] = load <4 x float>, ptr %v.addr
// CHECK-NEXT:    [[ELT:%.*]] = extractelement <4 x float> [[VEC]], i64 2
// CHECK-NEXT:    store float [[ELT]], ptr [[TMP]]
// CHECK-NEXT:    [[RES:%.*]] = load i32, ptr [[TMP]]
// CHECK-NEXT:    call void @llvm.lifetime.end.p0(ptr [[TMP]])
// CHECK-NEXT:    ret i32 [[RES]]
unsigned ext_vector_element(v4f v) {
  return __builtin_bit_cast(unsigned, v.z);
}

struct S {
  unsigned a : 4, b : 12;
};

// CHECK-LABEL: define{{.*}} i32 @bit_field(
// CHECK:         [[TMP:%ref.tmp.*]] = alloca i32
// CHECK:         call void @llvm.lifetime.start.p0(ptr [[TMP]])
// CHECK-NEXT:    [[LOAD:%.*]] = load i16, ptr %s
// CHECK-NEXT:    [[SHR:%.*]] = lshr i16 [[LOAD]], 4
// CHECK-NEXT:    [[EXT:%.*]] = zext i16 [[SHR]] to i32
// CHECK-NEXT:    store i32 [[EXT]], ptr [[TMP]]
// CHECK-NEXT:    [[RES:%.*]] = load i32, ptr [[TMP]]
// CHECK-NEXT:    call void @llvm.lifetime.end.p0(ptr [[TMP]])
// CHECK-NEXT:    ret i32 [[RES]]
unsigned bit_field(struct S s) {
  return __builtin_bit_cast(unsigned, s.b);
}

// CHECK-LABEL: define{{.*}} i32 @prvalue(
// CHECK:         [[TMP:%ref.tmp.*]] = alloca float
// CHECK:         call void @llvm.lifetime.start.p0(ptr [[TMP]])
// CHECK-NEXT:    [[F:%.*]] = load float, ptr %f.addr
// CHECK-NEXT:    [[ADD:%.*]] = fadd float [[F]], 1.000000e+00
// CHECK-NEXT:    store float [[ADD]], ptr [[TMP]]
// CHECK-NEXT:    [[RES:%.*]] = load i32, ptr [[TMP]]
// CHECK-NEXT:    call void @llvm.lifetime.end.p0(ptr [[TMP]])
// CHECK-NEXT:    ret i32 [[RES]]
unsigned prvalue(float f) {
  return __builtin_bit_cast(unsigned, f + 1.0f);
}

volatile float vf;

// CHECK-LABEL: define{{.*}} i32 @volatile_lvalue(
// CHECK-NOT:     alloca
// CHECK:         [[RES:%.*]] = load volatile i32, ptr @vf
// CHECK-NEXT:    ret i32 [[RES]]
unsigned volatile_lvalue(void) {
  return __builtin_bit_cast(unsigned, vf);
}
