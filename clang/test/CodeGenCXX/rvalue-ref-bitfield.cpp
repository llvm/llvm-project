// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-unknown \
// RUN:   -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

// static_cast<T&&> of a bit-field lvalue materializes a temporary holding the
// loaded bit-field value, and the reference binds to that temporary.

struct A {
  int a : 4;
};

// CHECK-LABEL: define {{.*}} @_Z4test1A(
void test(A obj) {
  // CHECK:      %[[OBJ:.*]] = alloca %struct.A
  // CHECK-NEXT: %[[RI:.*]] = alloca ptr
  // CHECK-NEXT: %[[TMP1:.*]] = alloca i32
  // CHECK-NEXT: %[[CRI:.*]] = alloca ptr
  // CHECK-NEXT: %[[TMP2:.*]] = alloca i32
  // CHECK-NEXT: store i32 %obj.coerce, ptr %[[OBJ]]

  // CHECK-NEXT: %[[LD1:.*]] = load i8, ptr %[[OBJ]]
  // CHECK-NEXT: %[[SHL1:.*]] = shl i8 %[[LD1]], 4
  // CHECK-NEXT: %[[ASHR1:.*]] = ashr i8 %[[SHL1]], 4
  // CHECK-NEXT: %[[CAST1:.*]] = sext i8 %[[ASHR1]] to i32
  // CHECK-NEXT: store i32 %[[CAST1]], ptr %[[TMP1]]
  // CHECK-NEXT: store ptr %[[TMP1]], ptr %[[RI]]
  int &&ri = static_cast<int&&>(obj.a);

  // CHECK-NEXT: %[[LD2:.*]] = load i8, ptr %[[OBJ]]
  // CHECK-NEXT: %[[SHL2:.*]] = shl i8 %[[LD2]], 4
  // CHECK-NEXT: %[[ASHR2:.*]] = ashr i8 %[[SHL2]], 4
  // CHECK-NEXT: %[[CAST2:.*]] = sext i8 %[[ASHR2]] to i32
  // CHECK-NEXT: store i32 %[[CAST2]], ptr %[[TMP2]]
  // CHECK-NEXT: store ptr %[[TMP2]], ptr %[[CRI]]
  const int &&cri = static_cast<const int &&>(obj.a);
}
