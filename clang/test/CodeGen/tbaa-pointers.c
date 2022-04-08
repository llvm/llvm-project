// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-apple-darwin -O1 -disable-llvm-passes %s -emit-llvm -o - | FileCheck %s

void p2unsigned(unsigned **ptr) {
  // CHECK-LABEL: define void @p2unsigned(i32** noundef %ptr)
  // CHECK-NEXT: entry:
  // CHECK-NEXT:  %ptr.addr = alloca i32**, align 8
  // CHECK-NEXT:  store i32** %ptr, i32*** %ptr.addr, align 8, !tbaa [[ANY_POINTER_0:!.+]]
  // CHECK-NEXT:  [[BASE:%.+]] = load i32**, i32*** %ptr.addr, align 8, !tbaa [[ANY_POINTER_0]]
  // CHECK-NEXT:  store i32* null, i32** [[BASE]], align 8, !tbaa [[ANY_POINTER_0]]
  // CHECK-NEXT:  ret void
  //
  *ptr = 0;
}

void p2unsigned_volatile(unsigned *volatile *ptr) {
  // CHECK-LABEL: define void @p2unsigned_volatile(i32** noundef %ptr)
  // CHECK-NEXT: entry:
  // CHECK-NEXT:   %ptr.addr = alloca i32**, align 8
  // CHECK-NEXT:   store i32** %ptr, i32*** %ptr.addr, align 8, !tbaa [[ANY_POINTER_0]]
  // CHECK-NEXT:   [[BASE:%.+]] = load i32**, i32*** %ptr.addr, align 8, !tbaa [[ANY_POINTER_0]]
  // CHECK-NEXT:   store volatile i32* null, i32** [[BASE]], align 8, !tbaa [[ANY_POINTER_0]]
  // CHECK-NEXT:   ret void
  //
  *ptr = 0;
}

void p3int(int ***ptr) {
  // CHECK-LABEL: define void @p3int(i32*** noundef %ptr)
  // CHECK-NEXT: entry:
  // CHECK-NEXT:   %ptr.addr = alloca i32***, align 8
  // CHECK-NEXT:   store i32*** %ptr, i32**** %ptr.addr, align 8, !tbaa [[ANY_POINTER_0]]
  // CHECK-NEXT:   [[BASE_0:%.+]] = load i32***, i32**** %ptr.addr, align 8, !tbaa [[ANY_POINTER_0]]
  // CHECK-NEXT:   [[BASE_1:%.+]] = load i32**, i32*** [[BASE_0]], align 8, !tbaa [[ANY_POINTER_0]]
  // CHECK-NEXT:   store i32* null, i32** [[BASE_1]], align 8, !tbaa [[ANY_POINTER_0]]
  // CHECK-NEXT:   ret void
  //
  **ptr = 0;
}

void p4char(char ****ptr) {
  // CHECK-LABEL: define void @p4char(i8**** noundef %ptr)
  // CHECK-NEXT: entry:
  // CHECK-NEXT:   %ptr.addr = alloca i8****, align 8
  // CHECK-NEXT:   store i8**** %ptr, i8***** %ptr.addr, align 8, !tbaa [[ANY_POINTER_0]]
  // CHECK-NEXT:   [[BASE_0:%.+]] = load i8****, i8***** %ptr.addr, align 8, !tbaa [[ANY_POINTER_0]]
  // CHECK-NEXT:   [[BASE_1:%.+]] = load i8***, i8**** [[BASE_0]], align 8, !tbaa [[ANY_POINTER_0]]
  // CHECK-NEXT:   [[BASE_2:%.+]] = load i8**, i8*** [[BASE_1]], align 8, !tbaa [[ANY_POINTER_0]]
  // CHECK-NEXT:   store i8* null, i8** [[BASE_2]], align 8, !tbaa [[ANY_POINTER_0]]
  // CHECK-NEXT:   ret void
  //
  ***ptr = 0;
}

void p4char_const1(const char ****ptr) {
  // CHECK-LABEL: define void @p4char_const1(i8**** noundef %ptr)
  // CHECK-NEXT: entry:
  // CHECK-NEXT:   %ptr.addr = alloca i8****, align 8
  // CHECK-NEXT:   store i8**** %ptr, i8***** %ptr.addr, align 8, !tbaa [[ANY_POINTER_0]]
  // CHECK-NEXT:   [[BASE_0:%.+]] = load i8****, i8***** %ptr.addr, align 8, !tbaa [[ANY_POINTER_0]]
  // CHECK-NEXT:   [[BASE_1:%.+]] = load i8***, i8**** [[BASE_0]], align 8, !tbaa [[ANY_POINTER_0]]
  // CHECK-NEXT:   [[BASE_2:%.+]] = load i8**, i8*** [[BASE_1]], align 8, !tbaa [[ANY_POINTER_0]]
  // CHECK-NEXT:   store i8* null, i8** [[BASE_2]], align 8, !tbaa [[ANY_POINTER_0]]
  // CHECK-NEXT:   ret void
  //
  ***ptr = 0;
}

void p4char_const2(const char **const **ptr) {
  // CHECK-LABEL: define void @p4char_const2(i8**** noundef %ptr)
  // CHECK-NEXT: entry:
  // CHECK-NEXT:   %ptr.addr = alloca i8****, align 8
  // CHECK-NEXT:   store i8**** %ptr, i8***** %ptr.addr, align 8, !tbaa [[ANY_POINTER_0]]
  // CHECK-NEXT:   [[BASE_0:%.+]] = load i8****, i8***** %ptr.addr, align 8, !tbaa [[ANY_POINTER_0]]
  // CHECK-NEXT:   [[BASE_1:%.+]] = load i8***, i8**** [[BASE_0]], align 8, !tbaa [[ANY_POINTER_0]]
  // CHECK-NEXT:   [[BASE_2:%.+]] = load i8**, i8*** [[BASE_1]], align 8, !tbaa [[ANY_POINTER_0]]
  // CHECK-NEXT:   store i8* null, i8** [[BASE_2]], align 8, !tbaa [[ANY_POINTER_0]]
  // CHECK-NEXT:   ret void
  //
  ***ptr = 0;
}

struct S1 {
  int x;
  int y;
};

void p2struct(struct S1 **ptr) {
  // CHECK-LABEL: define void @p2struct(%struct.S1** noundef %ptr)
  // CHECK-NEXT: entry:
  // CHECK-NEXT:   %ptr.addr = alloca %struct.S1**, align 8
  // CHECK-NEXT:   store %struct.S1** %ptr, %struct.S1*** %ptr.addr, align 8, !tbaa [[ANY_POINTER_0]]
  // CHECK-NEXT:   [[BASE:%.+]] = load %struct.S1**, %struct.S1*** %ptr.addr, align 8, !tbaa [[ANY_POINTER_0]]
  // CHECK-NEXT:   store %struct.S1* null, %struct.S1** [[BASE]], align 8, !tbaa [[ANY_POINTER_0]]
  // CHECK-NEXT:   ret void
  //
  *ptr = 0;
}

// CHECK: [[ANY_POINTER_0]] = !{[[ANY_POINTER:!.+]], [[ANY_POINTER]], i64 0}
// CHECK: [[ANY_POINTER]] = !{!"any pointer", [[CHAR:!.+]], i64 0}
// CHECK: [[CHAR]] = !{!"omnipotent char", [[TBAA_ROOT:!.+]], i64 0}
// CHECK: [[TBAA_ROOT]] = !{!"Simple C/C++ TBAA"}
//
