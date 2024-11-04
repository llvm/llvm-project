// RUN: %clang_cc1 -triple x86_64-apple-darwin -O1 -disable-llvm-passes %s -emit-llvm -o - | FileCheck --check-prefixes=COMMON,DEFAULT %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -O1 -disable-llvm-passes -pointer-tbaa %s -emit-llvm -o - | FileCheck --check-prefixes=COMMON,ENABLED %s
// RUN: %clang --target=x86_64-apple-darwin -O1 %s -emit-llvm -S -mllvm -disable-llvm-optzns -o - | FileCheck --check-prefixes=COMMON,DEFAULT %s
// RUN: %clang --target=x86_64-apple-darwin -O1 -fpointer-tbaa %s -emit-llvm -S -mllvm -disable-llvm-optzns -o - | FileCheck --check-prefixes=COMMON,ENABLED %s

void p2unsigned(unsigned **ptr) {
  // COMMON-LABEL: define void @p2unsigned(
  // COMMON-SAME:    ptr noundef [[PTR:%.+]])
  // COMMON:        [[PTR_ADDR:%.+]] = alloca ptr, align 8
  // ENABLED-NEXT:  store ptr [[PTR]], ptr [[PTR_ADDR]], align 8, !tbaa [[P2INT_0:!.+]]
  // ENABLED-NEXT:  [[BASE:%.+]] = load ptr, ptr [[PTR_ADDR]], align 8, !tbaa [[P2INT_0]]
  // ENABLED-NEXT:  store ptr null, ptr [[BASE]], align 8, !tbaa [[P1INT_0:!.+]]
  // DEFAULT-NEXT:  store ptr [[PTR]], ptr [[PTR_ADDR]], align 8, !tbaa [[ANYPTR:!.+]]
  // DEFAULT-NEXT:  [[BASE:%.+]] = load ptr, ptr [[PTR_ADDR]], align 8, !tbaa [[ANYPTR]]
  // DEFAULT-NEXT:  store ptr null, ptr [[BASE]], align 8, !tbaa [[ANYPTR]]
  // COMMON-NEXT:   ret void
  //
  *ptr = 0;
}

void p2unsigned_volatile(unsigned *volatile *ptr) {
  // COMMON-LABEL: define void @p2unsigned_volatile(
  // COMMON-SAME:    ptr noundef [[PTR:%.+]])
  // COMMON:         [[PTR_ADDR:%.+]] = alloca ptr, align 8
  // ENABLED-NEXT:   store ptr [[PTR]], ptr [[PTR_ADDR]], align 8, !tbaa [[P2INT_0]]
  // ENABLED-NEXT:   [[BASE:%.+]] = load ptr, ptr [[PTR_ADDR]], align 8, !tbaa [[P2INT_0]]
  // ENABLED-NEXT:   store volatile ptr null, ptr [[BASE]], align 8, !tbaa [[P1INT_0]]
  // DEFAULT-NEXT:   store ptr [[PTR]], ptr [[PTR_ADDR]], align 8, !tbaa [[ANYPTR]]
  // DEFAULT-NEXT:   [[BASE:%.+]] = load ptr, ptr [[PTR_ADDR]], align 8, !tbaa [[ANYPTR]]
  // DEFAULT-NEXT:   store volatile ptr null, ptr [[BASE]], align 8, !tbaa [[ANYPTR]]
  // COMMON-NEXT:    ret void
  //
  *ptr = 0;
}

void p3int(int ***ptr) {
  // COMMON-LABEL: define void @p3int(
  // COMMON-SAME:    ptr noundef [[PTR:%.+]])
  // COMMON:         [[PTR_ADDR:%.+]] = alloca ptr, align 8
  // ENABLED-NEXT:   store ptr [[PTR]], ptr [[PTR_ADDR]], align 8, !tbaa [[P3INT_0:!.+]]
  // ENABLED-NEXT:   [[BASE_0:%.+]] = load ptr, ptr [[PTR_ADDR]], align 8, !tbaa [[P3INT_0]]
  // ENABLED-NEXT:   [[BASE_1:%.+]] = load ptr, ptr [[BASE_0]], align 8, !tbaa [[P2INT_0]]
  // ENABLED-NEXT:   store ptr null, ptr [[BASE_1]], align 8, !tbaa [[P1INT_0]]
  // DEFAULT-NEXT:   store ptr [[PTR]], ptr [[PTR_ADDR]], align 8, !tbaa [[ANYPTR]]
  // DEFAULT-NEXT:   [[BASE_0:%.+]] = load ptr, ptr [[PTR_ADDR]], align 8, !tbaa [[ANYPTR]]
  // DEFAULT-NEXT:   [[BASE_1:%.+]] = load ptr, ptr [[BASE_0]], align 8, !tbaa [[ANYPTR]]
  // DEFAULT-NEXT:   store ptr null, ptr [[BASE_1]], align 8, !tbaa [[ANYPTR]]
  // COMMON-NEXT:    ret void
  //
  **ptr = 0;
}

void p4char(char ****ptr) {
  // COMMON-LABEL: define void @p4char(
  // COMMON-SAME:    ptr noundef [[PTR:%.+]])
  // COMMON:         [[PTR_ADDR:%.+]] = alloca ptr, align 8
  // ENABLED-NEXT:   store ptr [[PTR]], ptr [[PTR_ADDR]], align 8, !tbaa [[P4CHAR_0:!.+]]
  // ENABLED-NEXT:   [[BASE_0:%.+]] = load ptr, ptr [[PTR_ADDR]], align 8, !tbaa [[P4CHAR_0]]
  // ENABLED-NEXT:   [[BASE_1:%.+]] = load ptr, ptr [[BASE_0]], align 8, !tbaa [[P3CHAR_0:!.+]]
  // ENABLED-NEXT:   [[BASE_2:%.+]] = load ptr, ptr [[BASE_1]], align 8, !tbaa [[P2CHAR_0:!.+]]
  // ENABLED-NEXT:   store ptr null, ptr [[BASE_2]], align 8, !tbaa [[P1CHAR_0:!.+]]
  // DEFAULT-NEXT:   store ptr [[PTR]], ptr [[PTR_ADDR]], align 8, !tbaa [[ANYPTR]]
  // DEFAULT-NEXT:   [[BASE_0:%.+]] = load ptr, ptr [[PTR_ADDR]], align 8, !tbaa [[ANYPTR]]
  // DEFAULT-NEXT:   [[BASE_1:%.+]] = load ptr, ptr [[BASE_0]], align 8, !tbaa [[ANYPTR]]
  // DEFAULT-NEXT:   [[BASE_2:%.+]] = load ptr, ptr [[BASE_1]], align 8, !tbaa [[ANYPTR]]
  // DEFAULT-NEXT:   store ptr null, ptr [[BASE_2]], align 8, !tbaa [[ANYPTR]]
  // COMMON-NEXT:    ret void
  //
  ***ptr = 0;
}

void p4char_const1(const char ****ptr) {
  // COMMON-LABEL: define void @p4char_const1(
  // COMMON-SAME:    ptr noundef [[PTR:%.+]])
  // COMMON:         [[PTR_ADDR:%.+]] = alloca ptr, align 8
  // ENABLED-NEXT:   store ptr [[PTR]], ptr [[PTR_ADDR]], align 8, !tbaa [[P4CHAR_0]]
  // ENABLED-NEXT:   [[BASE_0:%.+]] = load ptr, ptr [[PTR_ADDR]], align 8, !tbaa [[P4CHAR_0]]
  // ENABLED-NEXT:   [[BASE_1:%.+]] = load ptr, ptr [[BASE_0]], align 8, !tbaa [[P3CHAR_0]]
  // ENABLED-NEXT:   [[BASE_2:%.+]] = load ptr, ptr [[BASE_1]], align 8, !tbaa [[P2CHAR_0]]
  // ENABLED-NEXT:   store ptr null, ptr [[BASE_2]], align 8, !tbaa [[P1CHAR_0]]
  // DEFAULT-NEXT:   store ptr [[PTR]], ptr [[PTR_ADDR]], align 8, !tbaa [[ANYPTR]]
  // DEFAULT-NEXT:   [[BASE_0:%.+]] = load ptr, ptr [[PTR_ADDR]], align 8, !tbaa [[ANYPTR]]
  // DEFAULT-NEXT:   [[BASE_1:%.+]] = load ptr, ptr [[BASE_0]], align 8, !tbaa [[ANYPTR]]
  // DEFAULT-NEXT:   [[BASE_2:%.+]] = load ptr, ptr [[BASE_1]], align 8, !tbaa [[ANYPTR]]
  // DEFAULT-NEXT:   store ptr null, ptr [[BASE_2]], align 8, !tbaa [[ANYPTR]]
  // COMMON-NEXT:   ret void
  //
  ***ptr = 0;
}

void p4char_const2(const char **const **ptr) {
  // COMMON-LABEL: define void @p4char_const2(
  // COMMON-SAME:    ptr noundef [[PTR:%.+]])
  // COMMON:         [[PTR_ADDR:%.+]] = alloca ptr, align 8
  // ENABLED-NEXT:   store ptr [[PTR]], ptr [[PTR_ADDR]], align 8, !tbaa [[P4CHAR_0]]
  // ENABLED-NEXT:   [[BASE_0:%.+]] = load ptr, ptr [[PTR_ADDR]], align 8, !tbaa [[P4CHAR_0]]
  // ENABLED-NEXT:   [[BASE_1:%.+]] = load ptr, ptr [[BASE_0]], align 8, !tbaa [[P3CHAR_0]]
  // ENABLED-NEXT:   [[BASE_2:%.+]] = load ptr, ptr [[BASE_1]], align 8, !tbaa [[P2CHAR_0]]
  // ENABLED-NEXT:   store ptr null, ptr [[BASE_2]], align 8, !tbaa [[P1CHAR_0]]
  // DEFAULT-NEXT:   store ptr [[PTR]], ptr [[PTR_ADDR]], align 8, !tbaa [[ANYPTR]]
  // DEFAULT-NEXT:   [[BASE_0:%.+]] = load ptr, ptr [[PTR_ADDR]], align 8, !tbaa [[ANYPTR]]
  // DEFAULT-NEXT:   [[BASE_1:%.+]] = load ptr, ptr [[BASE_0]], align 8, !tbaa [[ANYPTR]]
  // DEFAULT-NEXT:   [[BASE_2:%.+]] = load ptr, ptr [[BASE_1]], align 8, !tbaa [[ANYPTR]]
  // DEFAULT-NEXT:   store ptr null, ptr [[BASE_2]], align 8, !tbaa [[ANYPTR]]
  // COMMON-NEXT:    ret void
  //
  ***ptr = 0;
}

struct S1 {
  int x;
  int y;
};

void p2struct(struct S1 **ptr) {
  // COMMON-LABEL: define void @p2struct(
  // COMMON-SAME:    ptr noundef [[PTR:%.+]])
  // COMMON:         [[PTR_ADDR:%.+]] = alloca ptr, align 8
  // ENABLED-NEXT:   store ptr [[PTR]], ptr [[PTR_ADDR]], align 8, !tbaa [[P2S1_0:!.+]]
  // ENABLED-NEXT:   [[BASE:%.+]] = load ptr, ptr [[PTR_ADDR]], align 8, !tbaa [[P2S1_0]]
  // ENABLED-NEXT:   store ptr null, ptr [[BASE]], align 8, !tbaa [[P1S1_:!.+]]
  // DEFAULT-NEXT:   store ptr [[PTR]], ptr [[PTR_ADDR]], align 8, !tbaa [[ANYPTR]]
  // DEFAULT-NEXT:   [[BASE:%.+]] = load ptr, ptr [[PTR_ADDR]], align 8, !tbaa [[ANYPTR]]
  // DEFAULT-NEXT:   store ptr null, ptr [[BASE]], align 8, !tbaa [[ANYPTR]]
  // COMMON-NEXT:    ret void
  //
  *ptr = 0;
}

// ENABLED: [[P2INT_0]] = !{[[P2INT:!.+]], [[P2INT]], i64 0}
// ENABLED: [[P2INT]] = !{!"p2 int", [[ANY_POINTER:!.+]], i64 0}
// DEFAULT: [[ANYPTR]] = !{[[ANY_POINTER:!.+]], [[ANY_POINTER]], i64 0}
// COMMON: [[ANY_POINTER]] = !{!"any pointer", [[CHAR:!.+]], i64 0}
// COMMON: [[CHAR]] = !{!"omnipotent char", [[TBAA_ROOT:!.+]], i64 0}
// COMMON: [[TBAA_ROOT]] = !{!"Simple C/C++ TBAA"}
// ENABLED: [[P1INT_0]] = !{[[P1INT:!.+]], [[P1INT]], i64 0}
// ENABLED: [[P1INT]] = !{!"p1 int", [[ANY_POINTER]], i64 0}
// ENABLED: [[P3INT_0]] = !{[[P3INT:!.+]], [[P3INT]], i64 0}
// ENABLED: [[P3INT]] = !{!"p3 int", [[ANY_POINTER]], i64 0}
// ENABLED: [[P4CHAR_0]] = !{[[P4CHAR:!.+]], [[P4CHAR]], i64 0}
// ENABLED: [[P4CHAR]] = !{!"p4 omnipotent char", [[ANY_POINTER]], i64 0}
// ENABLED: [[P3CHAR_0]] = !{[[P3CHAR:!.+]], [[P3CHAR]], i64 0}
// ENABLED: [[P3CHAR]] = !{!"p3 omnipotent char", [[ANY_POINTER]], i64 0}
// ENABLED: [[P2CHAR_0]] = !{[[P2CHAR:!.+]], [[P2CHAR]], i64 0}
// ENABLED: [[P2CHAR]] = !{!"p2 omnipotent char", [[ANY_POINTER]], i64 0}
// ENABLED: [[P1CHAR_0]] = !{[[P1CHAR:!.+]], [[P1CHAR]], i64 0}
// ENABLED: [[P1CHAR]] = !{!"p1 omnipotent char", [[ANY_POINTER]], i64 0}
