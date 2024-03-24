// RUN: %clang_cc1 -triple x86_64-apple-darwin -O1 -disable-llvm-passes %s -emit-llvm -o - | FileCheck %s

void p2unsigned(unsigned **ptr) {
  // CHECK-LABEL: define void @p2unsigned(ptr noundef %ptr)
  // CHECK-NEXT: entry:
  // CHECK-NEXT:  %ptr.addr = alloca ptr, align 8
  // CHECK-NEXT:  store ptr %ptr, ptr %ptr.addr, align 8, !tbaa [[P2INT_0:!.+]]
  // CHECK-NEXT:  [[BASE:%.+]] = load ptr, ptr %ptr.addr, align 8, !tbaa [[P2INT_0]]
  // CHECK-NEXT:  store ptr null, ptr [[BASE]], align 8, !tbaa [[P1INT_0:!.+]]
  // CHECK-NEXT:  ret void
  //
  *ptr = 0;
}

void p2unsigned_volatile(unsigned *volatile *ptr) {
  // CHECK-LABEL: define void @p2unsigned_volatile(ptr noundef %ptr)
  // CHECK-NEXT: entry:
  // CHECK-NEXT:   %ptr.addr = alloca ptr, align 8
  // CHECK-NEXT:   store ptr %ptr, ptr %ptr.addr, align 8, !tbaa [[P2INT_0]]
  // CHECK-NEXT:   [[BASE:%.+]] = load ptr, ptr %ptr.addr, align 8, !tbaa [[P2INT_0]]
  // CHECK-NEXT:   store volatile ptr null, ptr [[BASE]], align 8, !tbaa [[P1INT_0]]
  // CHECK-NEXT:   ret void
  //
  *ptr = 0;
}

void p3int(int ***ptr) {
  // CHECK-LABEL: define void @p3int(ptr noundef %ptr)
  // CHECK-NEXT: entry:
  // CHECK-NEXT:   %ptr.addr = alloca ptr, align 8
  // CHECK-NEXT:   store ptr %ptr, ptr %ptr.addr, align 8, !tbaa [[P3INT_0:!.+]]
  // CHECK-NEXT:   [[BASE_0:%.+]] = load ptr, ptr %ptr.addr, align 8, !tbaa [[P3INT_0]]
  // CHECK-NEXT:   [[BASE_1:%.+]] = load ptr, ptr [[BASE_0]], align 8, !tbaa [[P2INT_0]]
  // CHECK-NEXT:   store ptr null, ptr [[BASE_1]], align 8, !tbaa [[P1INT_0]]
  // CHECK-NEXT:   ret void
  //
  **ptr = 0;
}

void p4char(char ****ptr) {
  // CHECK-LABEL: define void @p4char(ptr noundef %ptr)
  // CHECK-NEXT: entry:
  // CHECK-NEXT:   %ptr.addr = alloca ptr, align 8
  // CHECK-NEXT:   store ptr %ptr, ptr %ptr.addr, align 8, !tbaa [[P4CHAR_0:!.+]]
  // CHECK-NEXT:   [[BASE_0:%.+]] = load ptr, ptr %ptr.addr, align 8, !tbaa [[P4CHAR_0]]
  // CHECK-NEXT:   [[BASE_1:%.+]] = load ptr, ptr [[BASE_0]], align 8, !tbaa [[P3CHAR_0:!.+]]
  // CHECK-NEXT:   [[BASE_2:%.+]] = load ptr, ptr [[BASE_1]], align 8, !tbaa [[P2CHAR_0:!.+]]
  // CHECK-NEXT:   store ptr null, ptr [[BASE_2]], align 8, !tbaa [[P1CHAR_0:!.+]]
  // CHECK-NEXT:   ret void
  //
  ***ptr = 0;
}

void p4char_const1(const char ****ptr) {
  // CHECK-LABEL: define void @p4char_const1(ptr noundef %ptr)
  // CHECK-NEXT: entry:
  // CHECK-NEXT:   %ptr.addr = alloca ptr, align 8
  // CHECK-NEXT:   store ptr %ptr, ptr %ptr.addr, align 8, !tbaa [[P4CHAR_0]]
  // CHECK-NEXT:   [[BASE_0:%.+]] = load ptr, ptr %ptr.addr, align 8, !tbaa [[P4CHAR_0]]
  // CHECK-NEXT:   [[BASE_1:%.+]] = load ptr, ptr [[BASE_0]], align 8, !tbaa [[P3CHAR_0]]
  // CHECK-NEXT:   [[BASE_2:%.+]] = load ptr, ptr [[BASE_1]], align 8, !tbaa [[P2CHAR_0]]
  // CHECK-NEXT:   store ptr null, ptr [[BASE_2]], align 8, !tbaa [[P1CHAR_0]]
  // CHECK-NEXT:   ret void
  //
  ***ptr = 0;
}

void p4char_const2(const char **const **ptr) {
  // CHECK-LABEL: define void @p4char_const2(ptr noundef %ptr)
  // CHECK-NEXT: entry:
  // CHECK-NEXT:   %ptr.addr = alloca ptr, align 8
  // CHECK-NEXT:   store ptr %ptr, ptr %ptr.addr, align 8, !tbaa [[P4CHAR_0]]
  // CHECK-NEXT:   [[BASE_0:%.+]] = load ptr, ptr %ptr.addr, align 8, !tbaa [[P4CHAR_0]]
  // CHECK-NEXT:   [[BASE_1:%.+]] = load ptr, ptr [[BASE_0]], align 8, !tbaa [[P3CHAR_0]]
  // CHECK-NEXT:   [[BASE_2:%.+]] = load ptr, ptr [[BASE_1]], align 8, !tbaa [[P2CHAR_0]]
  // CHECK-NEXT:   store ptr null, ptr [[BASE_2]], align 8, !tbaa [[P1CHAR_0]]
  // CHECK-NEXT:   ret void
  //
  ***ptr = 0;
}

struct S1 {
  int x;
  int y;
};

void p2struct(struct S1 **ptr) {
  // CHECK-LABEL: define void @p2struct(ptr noundef %ptr)
  // CHECK-NEXT: entry:
  // CHECK-NEXT:   %ptr.addr = alloca ptr, align 8
  // CHECK-NEXT:   store ptr %ptr, ptr %ptr.addr, align 8, !tbaa [[P2S1_0:!.+]]
  // CHECK-NEXT:   [[BASE:%.+]] = load ptr, ptr %ptr.addr, align 8, !tbaa [[P2S1_0]]
  // CHECK-NEXT:   store ptr null, ptr [[BASE]], align 8, !tbaa [[P1S1_:!.+]]
  // CHECK-NEXT:   ret void
  //
  *ptr = 0;
}

// CHECK: [[P2INT_0]] = !{[[P2INT:!.+]], [[P2INT]], i64 0}
// CHECK: [[P2INT]] = !{!"p2 int", [[ANY_POINTER:!.+]], i64 0}
// CHECK: [[ANY_POINTER]] = !{!"any pointer", [[CHAR:!.+]], i64 0}
// CHECK: [[CHAR]] = !{!"omnipotent char", [[TBAA_ROOT:!.+]], i64 0}
// CHECK: [[TBAA_ROOT]] = !{!"Simple C/C++ TBAA"}
// CHECK: [[P1INT_0]] = !{[[P1INT:!.+]], [[P1INT]], i64 0}
// CHECK: [[P1INT]] = !{!"p1 int", [[ANY_POINTER]], i64 0}
// CHECK: [[P3INT_0]] = !{[[P3INT:!.+]], [[P3INT]], i64 0}
// CHECK: [[P3INT]] = !{!"p3 int", [[ANY_POINTER]], i64 0}
// CHECK: [[P4CHAR_0]] = !{[[P4CHAR:!.+]], [[P4CHAR]], i64 0}
// CHECK: [[P4CHAR]] = !{!"p4 omnipotent char", [[ANY_POINTER]], i64 0}
// CHECK: [[P3CHAR_0]] = !{[[P3CHAR:!.+]], [[P3CHAR]], i64 0}
// CHECK: [[P3CHAR]] = !{!"p3 omnipotent char", [[ANY_POINTER]], i64 0}
// CHECK: [[P2CHAR_0]] = !{[[P2CHAR:!.+]], [[P2CHAR]], i64 0}
// CHECK: [[P2CHAR]] = !{!"p2 omnipotent char", [[ANY_POINTER]], i64 0}
// CHECK: [[P1CHAR_0]] = !{[[P1CHAR:!.+]], [[P1CHAR]], i64 0}
// CHECK: [[P1CHAR]] = !{!"p1 omnipotent char", [[ANY_POINTER]], i64 0}
