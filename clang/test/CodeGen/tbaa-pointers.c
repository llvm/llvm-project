// RUN: %clang_cc1 -triple x86_64-apple-darwin -O1 -disable-llvm-passes %s -emit-llvm -o - | FileCheck %s --check-prefix=ENABLED
// RUN: %clang_cc1 -triple x86_64-apple-darwin -no-pointer-tbaa -O1 -disable-llvm-passes %s -emit-llvm -o - | FileCheck %s --check-prefix=DISABLED

void p2unsigned(unsigned **ptr) {
  // ENABLED-LABEL: define void @p2unsigned(ptr noundef %ptr)
  // ENABLED-NEXT: entry:
  // ENABLED-NEXT:  %ptr.addr = alloca ptr, align 8
  // ENABLED-NEXT:  store ptr %ptr, ptr %ptr.addr, align 8, !tbaa [[P2_INT:!.+]]
  // ENABLED-NEXT:  [[BASE:%.+]] = load ptr, ptr %ptr.addr, align 8, !tbaa [[P2_INT]]
  // ENABLED-NEXT:  store ptr null, ptr [[BASE]], align 8, !tbaa [[P1_INT:!.+]]
  // ENABLED-NEXT:  ret void
  //
  // DISABLED-LABEL: define void @p2unsigned(ptr noundef %ptr)
  // DISABLED-NEXT: entry:
  // DISABLED-NEXT:  %ptr.addr = alloca ptr, align 8
  // DISABLED-NEXT:  store ptr %ptr, ptr %ptr.addr, align 8, !tbaa [[ANY_POINTER:!.+]]
  // DISABLED-NEXT:  [[BASE:%.+]] = load ptr, ptr %ptr.addr, align 8, !tbaa [[ANY_POINTER]]
  // DISABLED-NEXT:  store ptr null, ptr [[BASE]], align 8, !tbaa [[ANY_POINTER]]
  // DISABLED-NEXT:  ret void
  //
  *ptr = 0;
}

void p2unsigned_volatile(unsigned *volatile *ptr) {
  // ENABLED-LABEL: define void @p2unsigned_volatile(ptr noundef %ptr)
  // ENABLED-NEXT: entry:
  // ENABLED-NEXT:   %ptr.addr = alloca ptr, align 8
  // ENABLED-NEXT:   store ptr %ptr, ptr %ptr.addr, align 8, !tbaa [[P2_INT]]
  // ENABLED-NEXT:   [[BASE:%.+]] = load ptr, ptr %ptr.addr, align 8, !tbaa [[P2_INT]]
  // ENABLED-NEXT:   store volatile ptr null, ptr [[BASE]], align 8, !tbaa [[P1_INT]]
  // ENABLED-NEXT:   ret void
  //
  // DISABLED-LABEL: define void @p2unsigned_volatile(ptr noundef %ptr)
  // DISABLED-NEXT: entry:
  // DISABLED-NEXT:   %ptr.addr = alloca ptr, align 8
  // DISABLED-NEXT:   store ptr %ptr, ptr %ptr.addr, align 8, !tbaa [[ANY_POINTER]]
  // DISABLED-NEXT:   [[BASE:%.+]] = load ptr, ptr %ptr.addr, align 8, !tbaa [[ANY_POINTER]]
  // DISABLED-NEXT:   store volatile ptr null, ptr [[BASE]], align 8, !tbaa [[ANY_POINTER]]
  // DISABLED-NEXT:   ret void
  //
  *ptr = 0;
}

void p3int(int ***ptr) {
  // ENABLED-LABEL: define void @p3int(ptr noundef %ptr)
  // ENABLED-NEXT: entry:
  // ENABLED-NEXT:   %ptr.addr = alloca ptr, align 8
  // ENABLED-NEXT:   store ptr %ptr, ptr %ptr.addr, align 8, !tbaa [[P3_INT:!.+]]
  // ENABLED-NEXT:   [[BASE_0:%.+]] = load ptr, ptr %ptr.addr, align 8, !tbaa [[P3_INT]]
  // ENABLED-NEXT:   [[BASE_1:%.+]] = load ptr, ptr [[BASE_0]], align 8, !tbaa [[P2_INT]]
  // ENABLED-NEXT:   store ptr null, ptr [[BASE_1]], align 8, !tbaa [[P1_INT]]
  // ENABLED-NEXT:   ret void
  //
  // DISABLED-LABEL: define void @p3int(ptr noundef %ptr)
  // DISABLED-NEXT: entry:
  // DISABLED-NEXT:   %ptr.addr = alloca ptr, align 8
  // DISABLED-NEXT:   store ptr %ptr, ptr %ptr.addr, align 8, !tbaa [[ANY_POINTER]]
  // DISABLED-NEXT:   [[BASE_0:%.+]] = load ptr, ptr %ptr.addr, align 8, !tbaa [[ANY_POINTER]]
  // DISABLED-NEXT:   [[BASE_1:%.+]] = load ptr, ptr [[BASE_0]], align 8, !tbaa [[ANY_POINTER]]
  // DISABLED-NEXT:   store ptr null, ptr [[BASE_1]], align 8, !tbaa [[ANY_POINTER]]
  // DISABLED-NEXT:   ret void
  //
  **ptr = 0;
}

void p4char(char ****ptr) {
  // ENABLED-LABEL: define void @p4char(ptr noundef %ptr)
  // ENABLED-NEXT: entry:
  // ENABLED-NEXT:   %ptr.addr = alloca ptr, align 8
  // ENABLED-NEXT:   store ptr %ptr, ptr %ptr.addr, align 8, !tbaa [[P4_CHAR:!.+]]
  // ENABLED-NEXT:   [[BASE_0:%.+]] = load ptr, ptr %ptr.addr, align 8, !tbaa [[P4_CHAR]]
  // ENABLED-NEXT:   [[BASE_1:%.+]] = load ptr, ptr [[BASE_0]], align 8, !tbaa [[P3_CHAR:!.+]]
  // ENABLED-NEXT:   [[BASE_2:%.+]] = load ptr, ptr [[BASE_1]], align 8, !tbaa [[P2_CHAR:!.+]]
  // ENABLED-NEXT:   store ptr null, ptr [[BASE_2]], align 8, !tbaa [[P1_CHAR:!.+]]
  // ENABLED-NEXT:   ret void
  //
  // DISABLED-LABEL: define void @p4char(ptr noundef %ptr)
  // DISABLED-NEXT: entry:
  // DISABLED-NEXT:   %ptr.addr = alloca ptr, align 8
  // DISABLED-NEXT:   store ptr %ptr, ptr %ptr.addr, align 8, !tbaa [[ANY_POINTER]]
  // DISABLED-NEXT:   [[BASE_0:%.+]] = load ptr, ptr %ptr.addr, align 8, !tbaa [[ANY_POINTER]]
  // DISABLED-NEXT:   [[BASE_1:%.+]] = load ptr, ptr [[BASE_0]], align 8, !tbaa [[ANY_POINTER]]
  // DISABLED-NEXT:   [[BASE_2:%.+]] = load ptr, ptr [[BASE_1]], align 8, !tbaa [[ANY_POINTER]]
  // DISABLED-NEXT:   store ptr null, ptr [[BASE_2]], align 8, !tbaa [[ANY_POINTER]]
  // DISABLED-NEXT:   ret void
  //
  ***ptr = 0;
}

void p4char_const1(const char ****ptr) {
  // ENABLED-LABEL: define void @p4char_const1(ptr noundef %ptr)
  // ENABLED-NEXT: entry:
  // ENABLED-NEXT:   %ptr.addr = alloca ptr, align 8
  // ENABLED-NEXT:   store ptr %ptr, ptr %ptr.addr, align 8, !tbaa [[P4_CHAR]]
  // ENABLED-NEXT:   [[BASE_0:%.+]] = load ptr, ptr %ptr.addr, align 8, !tbaa [[P4_CHAR]]
  // ENABLED-NEXT:   [[BASE_1:%.+]] = load ptr, ptr [[BASE_0]], align 8, !tbaa [[P3_CHAR]]
  // ENABLED-NEXT:   [[BASE_2:%.+]] = load ptr, ptr [[BASE_1]], align 8, !tbaa [[P2_CHAR]]
  // ENABLED-NEXT:   store ptr null, ptr [[BASE_2]], align 8, !tbaa [[P1_CHAR]]
  // ENABLED-NEXT:   ret void
  //
  // DISABLED-LABEL: define void @p4char_const1(ptr noundef %ptr)
  // DISABLED-NEXT: entry:
  // DISABLED-NEXT:   %ptr.addr = alloca ptr, align 8
  // DISABLED-NEXT:   store ptr %ptr, ptr %ptr.addr, align 8, !tbaa [[ANY_POINTER]]
  // DISABLED-NEXT:   [[BASE_0:%.+]] = load ptr, ptr %ptr.addr, align 8, !tbaa [[ANY_POINTER]]
  // DISABLED-NEXT:   [[BASE_1:%.+]] = load ptr, ptr [[BASE_0]], align 8, !tbaa [[ANY_POINTER]]
  // DISABLED-NEXT:   [[BASE_2:%.+]] = load ptr, ptr [[BASE_1]], align 8, !tbaa [[ANY_POINTER]]
  // DISABLED-NEXT:   store ptr null, ptr [[BASE_2]], align 8, !tbaa [[ANY_POINTER]]
  // DISABLED-NEXT:   ret void
  //
  ***ptr = 0;
}

void p4char_const2(const char **const **ptr) {
  // ENABLED-LABEL: define void @p4char_const2(ptr noundef %ptr)
  // ENABLED-NEXT: entry:
  // ENABLED-NEXT:   %ptr.addr = alloca ptr, align 8
  // ENABLED-NEXT:   store ptr %ptr, ptr %ptr.addr, align 8, !tbaa [[P4_CHAR]]
  // ENABLED-NEXT:   [[BASE_0:%.+]] = load ptr, ptr %ptr.addr, align 8, !tbaa [[P4_CHAR]]
  // ENABLED-NEXT:   [[BASE_1:%.+]] = load ptr, ptr [[BASE_0]], align 8, !tbaa [[P3_CHAR]]
  // ENABLED-NEXT:   [[BASE_2:%.+]] = load ptr, ptr [[BASE_1]], align 8, !tbaa [[P2_CHAR]]
  // ENABLED-NEXT:   store ptr null, ptr [[BASE_2]], align 8, !tbaa [[P1_CHAR]]
  // ENABLED-NEXT:   ret void
  //
  // DISABLED-LABEL: define void @p4char_const2(ptr noundef %ptr)
  // DISABLED-NEXT: entry:
  // DISABLED-NEXT:   %ptr.addr = alloca ptr, align 8
  // DISABLED-NEXT:   store ptr %ptr, ptr %ptr.addr, align 8, !tbaa [[ANY_POINTER]]
  // DISABLED-NEXT:   [[BASE_0:%.+]] = load ptr, ptr %ptr.addr, align 8, !tbaa [[ANY_POINTER]]
  // DISABLED-NEXT:   [[BASE_1:%.+]] = load ptr, ptr [[BASE_0]], align 8, !tbaa [[ANY_POINTER]]
  // DISABLED-NEXT:   [[BASE_2:%.+]] = load ptr, ptr [[BASE_1]], align 8, !tbaa [[ANY_POINTER]]
  // DISABLED-NEXT:   store ptr null, ptr [[BASE_2]], align 8, !tbaa [[ANY_POINTER]]
  // DISABLED-NEXT:   ret void
  //
  ***ptr = 0;
}

struct S1 {
  int x;
  int y;
};

void p2struct(struct S1 **ptr) {
  // ENABLED-LABEL: define void @p2struct(ptr noundef %ptr)
  // ENABLED-NEXT: entry:
  // ENABLED-NEXT:   %ptr.addr = alloca ptr, align 8
  // ENABLED-NEXT:   store ptr %ptr, ptr %ptr.addr, align 8, !tbaa [[P2_S:!.+]]
  // ENABLED-NEXT:   [[BASE:%.+]] = load ptr, ptr %ptr.addr, align 8, !tbaa [[P2_S]]
  // ENABLED-NEXT:   store ptr null, ptr [[BASE]], align 8, !tbaa [[P1_S:!.+]]
  // ENABLED-NEXT:   ret void
  //
  // DISABLED-LABEL: define void @p2struct(ptr noundef %ptr)
  // DISABLED-NEXT: entry:
  // DISABLED-NEXT:   %ptr.addr = alloca ptr, align 8
  // DISABLED-NEXT:   store ptr %ptr, ptr %ptr.addr, align 8, !tbaa [[ANY_POINTER]]
  // DISABLED-NEXT:   [[BASE:%.+]] = load ptr, ptr %ptr.addr, align 8, !tbaa [[ANY_POINTER]]
  // DISABLED-NEXT:   store ptr null, ptr [[BASE]], align 8, !tbaa [[ANY_POINTER]]
  // DISABLED-NEXT:   ret void
  //
  *ptr = 0;
}

// ENABLED: [[P2_INT]] = !{[[P2_INT_TAG:!.+]], [[P2_INT_TAG]], i64 0}
// ENABLED: [[P2_INT_TAG]] = !{!"p2 int", [[ANY_POINTER_TAG:!.+]], i64 0}
// ENABLED: [[ANY_POINTER_TAG]] = !{!"any pointer", [[CHAR:!.+]], i64 0}
// ENABLED: [[CHAR]] = !{!"omnipotent char", [[TBAA_ROOT:!.+]], i64 0}
// ENABLED: [[TBAA_ROOT]] = !{!"Simple C/C++ TBAA"}
// ENABLED: [[P1_INT]] = !{[[P1_INT_TAG:!.+]], [[P1_INT_TAG]], i64 0}
// ENABLED: [[P1_INT_TAG]] = !{!"p1 int", [[ANY_POINTER_TAG]], i64 0}
// ENABLED: [[P4_CHAR]] = !{[[P4_CHAR_TAG:!.+]], [[P4_CHAR_TAG]], i64 0}
// ENABLED: [[P4_CHAR_TAG]] = !{!"p4 omnipotent char", [[ANY_POINTER_TAG]], i64 0}
// ENABLED: [[P3_CHAR]] = !{[[P3_CHAR_TAG:!.+]], [[P3_CHAR_TAG]], i64 0}
// ENABLED: [[P3_CHAR_TAG]] = !{!"p3 omnipotent char", [[ANY_POINTER_TAG]], i64 0}
// ENABLED: [[P2_CHAR]] = !{[[P2_CHAR_TAG:!.+]], [[P2_CHAR_TAG]], i64 0}
// ENABLED: [[P2_CHAR_TAG]] = !{!"p2 omnipotent char", [[ANY_POINTER_TAG]], i64 0}
// ENABLED: [[P1_CHAR]] = !{[[P1_CHAR_TAG:!.+]], [[P1_CHAR_TAG]], i64 0}
// ENABLED: [[P1_CHAR_TAG]] = !{!"p1 omnipotent char", [[ANY_POINTER_TAG]], i64 0}
// ENABLED: [[P2_S]] = !{[[P2_S_TAG:!.+]], [[P2_S_TAG]], i64 0}
// ENABLED: [[P2_S_TAG]] = !{!"p2 struct S1", [[ANY_POINTER_TAG]], i64 0}
// ENABLED: [[P1_S]] = !{[[P1_S_TAG:!.+]], [[P1_S_TAG]], i64 0}
// ENABLED: [[P1_S_TAG]] = !{!"p1 struct S1", [[ANY_POINTER_TAG]], i64 0}
//
// DISABLED: [[ANY_POINTER]] = !{[[ANY_POINTER_MD:!.+]], [[ANY_POINTER_MD]], i64 0}
// DISABLED: [[ANY_POINTER_MD]] = !{!"any pointer", [[CHAR:!.+]], i64 0}
// DISABLED: [[CHAR]] = !{!"omnipotent char", [[TBAA_ROOT:!.+]], i64 0}
// DISABLED: [[TBAA_ROOT]] = !{!"Simple C/C++ TBAA"}
//
