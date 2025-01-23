// RUN: %clang_cc1 -triple x86_64-apple-darwin -O1 -disable-llvm-passes -no-pointer-tbaa %s -emit-llvm -o - | FileCheck --check-prefixes=COMMON,DISABLE %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin -O1 -disable-llvm-passes %s -emit-llvm -o - | FileCheck --check-prefixes=COMMON,DEFAULT %s
// RUN: %clang --target=x86_64-apple-darwin -O1 -fno-pointer-tbaa %s -emit-llvm -S -mllvm -disable-llvm-optzns -o - | FileCheck --check-prefixes=COMMON,DISABLE %s
// RUN: %clang --target=x86_64-apple-darwin -O1 %s -emit-llvm -S -mllvm -disable-llvm-optzns -o - | FileCheck --check-prefixes=COMMON,DEFAULT %s

void p2unsigned(unsigned **ptr) {
  // COMMON-LABEL: define void @p2unsigned(
  // COMMON-SAME:    ptr noundef [[PTR:%.+]])
  // COMMON:        [[PTR_ADDR:%.+]] = alloca ptr, align 8
  // DEFAULT-NEXT:  store ptr [[PTR]], ptr [[PTR_ADDR]], align 8, !tbaa [[P2INT_0:!.+]]
  // DEFAULT-NEXT:  [[BASE:%.+]] = load ptr, ptr [[PTR_ADDR]], align 8, !tbaa [[P2INT_0]]
  // DEFAULT-NEXT:  store ptr null, ptr [[BASE]], align 8, !tbaa [[P1INT_0:!.+]]
  // DISABLE-NEXT:  store ptr [[PTR]], ptr [[PTR_ADDR]], align 8, !tbaa [[ANYPTR:!.+]]
  // DISABLE-NEXT:  [[BASE:%.+]] = load ptr, ptr [[PTR_ADDR]], align 8, !tbaa [[ANYPTR]]
  // DISABLE-NEXT:  store ptr null, ptr [[BASE]], align 8, !tbaa [[ANYPTR]]
  // COMMON-NEXT:   ret void
  //
  *ptr = 0;
}

void p2unsigned_volatile(unsigned *volatile *ptr) {
  // COMMON-LABEL: define void @p2unsigned_volatile(
  // COMMON-SAME:    ptr noundef [[PTR:%.+]])
  // COMMON:         [[PTR_ADDR:%.+]] = alloca ptr, align 8
  // DEFAULT-NEXT:   store ptr [[PTR]], ptr [[PTR_ADDR]], align 8, !tbaa [[P2INT_0]]
  // DEFAULT-NEXT:   [[BASE:%.+]] = load ptr, ptr [[PTR_ADDR]], align 8, !tbaa [[P2INT_0]]
  // DEFAULT-NEXT:   store volatile ptr null, ptr [[BASE]], align 8, !tbaa [[P1INT_0]]
  // DISABLE-NEXT:   store ptr [[PTR]], ptr [[PTR_ADDR]], align 8, !tbaa [[ANYPTR]]
  // DISABLE-NEXT:   [[BASE:%.+]] = load ptr, ptr [[PTR_ADDR]], align 8, !tbaa [[ANYPTR]]
  // DISABLE-NEXT:   store volatile ptr null, ptr [[BASE]], align 8, !tbaa [[ANYPTR]]
  // COMMON-NEXT:    ret void
  //
  *ptr = 0;
}

void p3int(int ***ptr) {
  // COMMON-LABEL: define void @p3int(
  // COMMON-SAME:    ptr noundef [[PTR:%.+]])
  // COMMON:         [[PTR_ADDR:%.+]] = alloca ptr, align 8
  // DEFAULT-NEXT:   store ptr [[PTR]], ptr [[PTR_ADDR]], align 8, !tbaa [[P3INT_0:!.+]]
  // DEFAULT-NEXT:   [[BASE_0:%.+]] = load ptr, ptr [[PTR_ADDR]], align 8, !tbaa [[P3INT_0]]
  // DEFAULT-NEXT:   [[BASE_1:%.+]] = load ptr, ptr [[BASE_0]], align 8, !tbaa [[P2INT_0]]
  // DEFAULT-NEXT:   store ptr null, ptr [[BASE_1]], align 8, !tbaa [[P1INT_0]]
  // DISABLE-NEXT:   store ptr [[PTR]], ptr [[PTR_ADDR]], align 8, !tbaa [[ANYPTR]]
  // DISABLE-NEXT:   [[BASE_0:%.+]] = load ptr, ptr [[PTR_ADDR]], align 8, !tbaa [[ANYPTR]]
  // DISABLE-NEXT:   [[BASE_1:%.+]] = load ptr, ptr [[BASE_0]], align 8, !tbaa [[ANYPTR]]
  // DISABLE-NEXT:   store ptr null, ptr [[BASE_1]], align 8, !tbaa [[ANYPTR]]
  // COMMON-NEXT:    ret void
  //
  **ptr = 0;
}

void p4char(char ****ptr) {
  // COMMON-LABEL: define void @p4char(
  // COMMON-SAME:    ptr noundef [[PTR:%.+]])
  // COMMON:         [[PTR_ADDR:%.+]] = alloca ptr, align 8
  // DEFAULT-NEXT:   store ptr [[PTR]], ptr [[PTR_ADDR]], align 8, !tbaa [[P4CHAR_0:!.+]]
  // DEFAULT-NEXT:   [[BASE_0:%.+]] = load ptr, ptr [[PTR_ADDR]], align 8, !tbaa [[P4CHAR_0]]
  // DEFAULT-NEXT:   [[BASE_1:%.+]] = load ptr, ptr [[BASE_0]], align 8, !tbaa [[P3CHAR_0:!.+]]
  // DEFAULT-NEXT:   [[BASE_2:%.+]] = load ptr, ptr [[BASE_1]], align 8, !tbaa [[P2CHAR_0:!.+]]
  // DEFAULT-NEXT:   store ptr null, ptr [[BASE_2]], align 8, !tbaa [[P1CHAR_0:!.+]]
  // DISABLE-NEXT:   store ptr [[PTR]], ptr [[PTR_ADDR]], align 8, !tbaa [[ANYPTR]]
  // DISABLE-NEXT:   [[BASE_0:%.+]] = load ptr, ptr [[PTR_ADDR]], align 8, !tbaa [[ANYPTR]]
  // DISABLE-NEXT:   [[BASE_1:%.+]] = load ptr, ptr [[BASE_0]], align 8, !tbaa [[ANYPTR]]
  // DISABLE-NEXT:   [[BASE_2:%.+]] = load ptr, ptr [[BASE_1]], align 8, !tbaa [[ANYPTR]]
  // DISABLE-NEXT:   store ptr null, ptr [[BASE_2]], align 8, !tbaa [[ANYPTR]]
  // COMMON-NEXT:    ret void
  //
  ***ptr = 0;
}

void p4char_const1(const char ****ptr) {
  // COMMON-LABEL: define void @p4char_const1(
  // COMMON-SAME:    ptr noundef [[PTR:%.+]])
  // COMMON:         [[PTR_ADDR:%.+]] = alloca ptr, align 8
  // DEFAULT-NEXT:   store ptr [[PTR]], ptr [[PTR_ADDR]], align 8, !tbaa [[P4CHAR_0]]
  // DEFAULT-NEXT:   [[BASE_0:%.+]] = load ptr, ptr [[PTR_ADDR]], align 8, !tbaa [[P4CHAR_0]]
  // DEFAULT-NEXT:   [[BASE_1:%.+]] = load ptr, ptr [[BASE_0]], align 8, !tbaa [[P3CHAR_0]]
  // DEFAULT-NEXT:   [[BASE_2:%.+]] = load ptr, ptr [[BASE_1]], align 8, !tbaa [[P2CHAR_0]]
  // DEFAULT-NEXT:   store ptr null, ptr [[BASE_2]], align 8, !tbaa [[P1CHAR_0]]
  // DISABLE-NEXT:   store ptr [[PTR]], ptr [[PTR_ADDR]], align 8, !tbaa [[ANYPTR]]
  // DISABLE-NEXT:   [[BASE_0:%.+]] = load ptr, ptr [[PTR_ADDR]], align 8, !tbaa [[ANYPTR]]
  // DISABLE-NEXT:   [[BASE_1:%.+]] = load ptr, ptr [[BASE_0]], align 8, !tbaa [[ANYPTR]]
  // DISABLE-NEXT:   [[BASE_2:%.+]] = load ptr, ptr [[BASE_1]], align 8, !tbaa [[ANYPTR]]
  // DISABLE-NEXT:   store ptr null, ptr [[BASE_2]], align 8, !tbaa [[ANYPTR]]
  // COMMON-NEXT:   ret void
  //
  ***ptr = 0;
}

void p4char_const2(const char **const **ptr) {
  // COMMON-LABEL: define void @p4char_const2(
  // COMMON-SAME:    ptr noundef [[PTR:%.+]])
  // COMMON:         [[PTR_ADDR:%.+]] = alloca ptr, align 8
  // DEFAULT-NEXT:   store ptr [[PTR]], ptr [[PTR_ADDR]], align 8, !tbaa [[P4CHAR_0]]
  // DEFAULT-NEXT:   [[BASE_0:%.+]] = load ptr, ptr [[PTR_ADDR]], align 8, !tbaa [[P4CHAR_0]]
  // DEFAULT-NEXT:   [[BASE_1:%.+]] = load ptr, ptr [[BASE_0]], align 8, !tbaa [[P3CHAR_0]]
  // DEFAULT-NEXT:   [[BASE_2:%.+]] = load ptr, ptr [[BASE_1]], align 8, !tbaa [[P2CHAR_0]]
  // DEFAULT-NEXT:   store ptr null, ptr [[BASE_2]], align 8, !tbaa [[P1CHAR_0]]
  // DISABLE-NEXT:   store ptr [[PTR]], ptr [[PTR_ADDR]], align 8, !tbaa [[ANYPTR]]
  // DISABLE-NEXT:   [[BASE_0:%.+]] = load ptr, ptr [[PTR_ADDR]], align 8, !tbaa [[ANYPTR]]
  // DISABLE-NEXT:   [[BASE_1:%.+]] = load ptr, ptr [[BASE_0]], align 8, !tbaa [[ANYPTR]]
  // DISABLE-NEXT:   [[BASE_2:%.+]] = load ptr, ptr [[BASE_1]], align 8, !tbaa [[ANYPTR]]
  // DISABLE-NEXT:   store ptr null, ptr [[BASE_2]], align 8, !tbaa [[ANYPTR]]
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
  // DEFAULT-NEXT:    store ptr [[PTR]], ptr [[PTR_ADDR]], align 8, !tbaa [[P2S1_TAG:!.+]]
  // DISABLE-NEXT:    store ptr [[PTR]], ptr [[PTR_ADDR]], align 8, !tbaa [[ANYPTR]]
  // DEFAULT-NEXT:    [[BASE:%.+]] = load ptr, ptr [[PTR_ADDR]], align 8, !tbaa [[P2S1_TAG]]
  // DISABLE-NEXT:    [[BASE:%.+]] = load ptr, ptr [[PTR_ADDR]], align 8, !tbaa [[ANYPTR]]
  // DEFAULT-NEXT:    store ptr null, ptr [[BASE]], align 8, !tbaa [[P1S1_TAG:!.+]]
  // DISABLE-NEXT:    store ptr null, ptr [[BASE]], align 8, !tbaa [[ANYPTR]]
  // COMMON-NEXT:    ret void
  //
  *ptr = 0;
}

void p2struct_const(struct S1 const **ptr) {
  // COMMON-LABEL: define void @p2struct_const(
  // COMMON-SAME:    ptr noundef [[PTR:%.+]])
  // COMMON:         [[PTR_ADDR:%.+]] = alloca ptr, align 8
  // COMMON-NEXT:    store ptr [[PTR]], ptr [[PTR_ADDR]], align 8, !tbaa [[ANYPTR:!.+]]
  // COMMON-NEXT:    [[BASE:%.+]] = load ptr, ptr [[PTR_ADDR]], align 8, !tbaa [[ANYPTR]]
  // DEFAULT-NEXT:    store ptr null, ptr [[BASE]], align 8, !tbaa [[P1S1_TAG]]
  // DISABLE-NEXT:    store ptr null, ptr [[BASE]], align 8, !tbaa [[ANYPTR]]
  // COMMON-NEXT:    ret void
  //
  *ptr = 0;
}

struct S2 {
  struct S1 *s;
};

void p2struct2(struct S2 *ptr) {
  // COMMON-LABEL: define void @p2struct2(
  // COMMON-SAME:    ptr noundef [[PTR:%.+]])
  // COMMON:         [[PTR_ADDR:%.+]] = alloca ptr, align 8
  // DEFAULT-NEXT:   store ptr [[PTR]], ptr [[PTR_ADDR]], align 8, !tbaa [[P1S2_TAG:!.+]]
  // DEFAULT-NEXT:   [[BASE:%.+]] = load ptr, ptr [[PTR_ADDR]], align 8, !tbaa [[P1S2_TAG]]
  // DEFAULT-NEXT:   [[S:%.+]] = getelementptr inbounds nuw %struct.S2, ptr [[BASE]], i32 0, i32 0
  // DEFAULT-NEXT:   store ptr null, ptr [[S]], align 8, !tbaa [[S2_S_TAG:!.+]]
  // DISABLE-NEXT:   store ptr [[PTR]], ptr [[PTR_ADDR]], align 8, !tbaa [[ANYPTR]]
  // DISABLE-NEXT:   [[BASE:%.+]] = load ptr, ptr [[PTR_ADDR]], align 8, !tbaa [[ANYPTR]]
  // DISABLE-NEXT:   [[S:%.+]] = getelementptr inbounds nuw %struct.S2, ptr [[BASE]], i32 0, i32 0
  // DISABLE-NEXT:   store ptr null, ptr [[S]], align 8, !tbaa [[S2_S_TAG:!.+]]
  // COMMON-NEXT:    ret void
    ptr->s = 0;
}


void vla1(int n, int ptr[][n], int idx) {
// COMMON-LABEL: define void @vla1(
// COMMON-SAME:    i32 noundef [[N:%.+]], ptr noundef [[PTR:%.+]], i32 noundef [[IDX:%.+]])
// COMMON:  	 [[N_ADDR:%.+]] = alloca i32, align 4
// COMMON-NEXT:  [[PTR_ADDR:%.+]] = alloca ptr, align 8
// COMMON-NEXT:  [[IDX_ADDR:%.+]] = alloca i32, align 4
// COMMON-NEXT: store i32 [[N]], ptr [[N_ADDR]], align 4, !tbaa [[INT_TY:!.+]]
// DEFAULT-NEXT: store ptr [[PTR]], ptr [[PTR_ADDR]], align 8, !tbaa [[P1INT0:!.+]]
// DISABLE-NEXT: store ptr [[PTR]], ptr [[PTR_ADDR]], align 8, !tbaa [[ANYPTR]]
// COMMON-NEXT: store i32 [[IDX]], ptr [[IDX_ADDR]], align 4, !tbaa [[INT_TY]]
// COMMON-NEXT: [[L:%.+]] = load i32, ptr [[N_ADDR]], align 4, !tbaa [[INT_TY]]
// COMMON-NEXT: [[L_EXT:%.+]] = zext i32 [[L]] to i64
// DEFAULT-NEXT: [[L_PTR:%.+]] = load ptr, ptr [[PTR_ADDR]], align 8, !tbaa [[P1INT0]]
// DISABLE-NEXT: [[L_PTR:%.+]] = load ptr, ptr [[PTR_ADDR]], align 8, !tbaa [[ANYPTR]]
// COMMON-NEXT: [[L_IDX:%.+]] = load i32, ptr [[IDX_ADDR]], align 4, !tbaa [[INT_TY]]
// COMMON-NEXT: [[IDX_EXT:%.+]] = sext i32 [[L_IDX]] to i64
// COMMON-NEXT: [[MUL:%.+]] = mul nsw i64 [[IDX_EXT]], [[L_EXT]]
// COMMON-NEXT: [[GEP1:%.+]] = getelementptr inbounds i32, ptr [[L_PTR]], i64 [[MUL]]
// COMMON-NEXT: [[GEP2:%.+]] = getelementptr inbounds i32, ptr [[GEP1]], i64 0
// COMMON-NEXT: store i32 0, ptr [[GEP2]], align 4, !tbaa [[INT_TAG:!.+]]
// DEFAULT-NEXT: ret void

    ptr[idx][0] = 0;
}

typedef struct {
  int i1;
} TypedefS;

void unamed_struct_typedef(TypedefS *ptr) {
// COMMON-LABEL: define void @unamed_struct_typedef(
// COMMON-SAME: ptr noundef [[PTRA:%.+]])
// COMMON:        [[PTR_ADDR:%.+]]  = alloca ptr, align 8
// DISABLE-NEXT:  store ptr [[PTRA]], ptr [[PTR_ADDR]], align 8, !tbaa [[ANYPTR]]
// DEFAULT-NEXT:  store ptr [[PTRA]], ptr [[PTR_ADDR]], align 8, !tbaa [[ANYPTR:!.+]]
// COMMON-NEXT:   [[L0:%.+]] = load ptr, ptr [[PTR_ADDR]], align 8, !tbaa  [[ANYPTR]]
// COMMON-NEXT:   [[GEP:%.+]]  = getelementptr inbounds nuw %struct.TypedefS, ptr [[L0]], i32 0, i32 0
// COMMON-NEXT:   store i32 0, ptr [[GEP]], align 4
// COMMON-NEXT:   ret void

  ptr->i1 = 0;
}

int void_ptrs(void **ptr) {
// COMMON-LABEL: define i32 @void_ptrs(
// COMMON-SAME: ptr noundef [[PTRA:%.+]])
// COMMON:        [[PTR_ADDR:%.+]]  = alloca ptr, align 8
// DISABLE-NEXT:  store ptr [[PTRA]], ptr [[PTR_ADDR]], align 8, !tbaa [[ANYPTR]]
// DISABLE-NEXT:  [[L0:%.+]] = load ptr, ptr  [[PTR_ADDR]], align 8, !tbaa  [[ANYPTR]]
// DISABLE-NEXT:  [[L1:%.+]] = load ptr, ptr [[L0]], align 8, !tbaa  [[ANYPTR]]
// DEFAULT-NEXT:  store ptr [[PTRA]], ptr [[PTR_ADDR]], align 8, !tbaa [[P2VOID:!.+]]
// DEFAULT-NEXT:  [[L0:%.+]] = load ptr, ptr  [[PTR_ADDR]], align 8, !tbaa  [[P2VOID]]
// DEFAULT-NEXT:  [[L1:%.+]] = load ptr, ptr [[L0]], align 8, !tbaa  [[P1VOID:!.+]]
// COMMON-NEXT:   [[BOOL:%.+]] = icmp ne ptr [[L1]], null
// COMMON-NEXT:   [[BOOL_EXT:%.+]] = zext i1 [[BOOL]] to i64
// COMMON-NEXT:   [[COND:%.+]] = select i1 [[BOOL]], i32 0, i32 1
// COMMON-NEXT:   ret i32 [[COND]]

  return *ptr ? 0 : 1;
}

// DEFAULT: [[P2INT_0]] = !{[[P2INT:!.+]], [[P2INT]], i64 0}
// DEFAULT: [[P2INT]] = !{!"p2 int", [[ANY_POINTER:!.+]], i64 0}
// DISABLE: [[ANYPTR]] = !{[[ANY_POINTER:!.+]], [[ANY_POINTER]], i64 0}
// COMMON: [[ANY_POINTER]] = !{!"any pointer", [[CHAR:!.+]], i64 0}
// COMMON: [[CHAR]] = !{!"omnipotent char", [[TBAA_ROOT:!.+]], i64 0}
// COMMON: [[TBAA_ROOT]] = !{!"Simple C/C++ TBAA"}
// DEFAULT: [[P1INT_0]] = !{[[P1INT:!.+]], [[P1INT]], i64 0}
// DEFAULT: [[P1INT]] = !{!"p1 int", [[ANY_POINTER]], i64 0}
// DEFAULT: [[P3INT_0]] = !{[[P3INT:!.+]], [[P3INT]], i64 0}
// DEFAULT: [[P3INT]] = !{!"p3 int", [[ANY_POINTER]], i64 0}
// DEFAULT: [[P4CHAR_0]] = !{[[P4CHAR:!.+]], [[P4CHAR]], i64 0}
// DEFAULT: [[P4CHAR]] = !{!"p4 omnipotent char", [[ANY_POINTER]], i64 0}
// DEFAULT: [[P3CHAR_0]] = !{[[P3CHAR:!.+]], [[P3CHAR]], i64 0}
// DEFAULT: [[P3CHAR]] = !{!"p3 omnipotent char", [[ANY_POINTER]], i64 0}
// DEFAULT: [[P2CHAR_0]] = !{[[P2CHAR:!.+]], [[P2CHAR]], i64 0}
// DEFAULT: [[P2CHAR]] = !{!"p2 omnipotent char", [[ANY_POINTER]], i64 0}
// DEFAULT: [[P1CHAR_0]] = !{[[P1CHAR:!.+]], [[P1CHAR]], i64 0}
// DEFAULT: [[P1CHAR]] = !{!"p1 omnipotent char", [[ANY_POINTER]], i64 0}
// DEFAULT: [[P2S1_TAG]] = !{[[P2S1:!.+]], [[P2S1]], i64 0}
// DEFAULT: [[P2S1]] = !{!"p2 _ZTS2S1", [[ANY_POINTER]], i64 0}
// DEFAULT: [[P1S1_TAG:!.+]] = !{[[P1S1:!.+]], [[P1S1]], i64 0}
// DEFAULT: [[P1S1]] = !{!"p1 _ZTS2S1", [[ANY_POINTER]], i64 0}
// DEFAULT: [[P1S2_TAG]] = !{[[P1S2:!.+]], [[P1S2]], i64 0}
// DEFAULT: [[P1S2]] = !{!"p1 _ZTS2S2", [[ANY_POINTER]], i64 0}

// DEFAULT: [[S2_S_TAG]]  = !{[[S2_TY:!.+]], [[P1S1]], i64 0}
// DEFAULT: [[S2_TY]]  = !{!"S2", [[P1S1]], i64 0}
// DISABLE: [[S2_S_TAG]]  = !{[[S2_TY:!.+]], [[ANY_POINTER]], i64 0}
// DISABLE: [[S2_TY]]  = !{!"S2", [[ANY_POINTER]], i64 0}
// COMMON:  [[INT_TAG]] = !{[[INT_TY:!.+]], [[INT_TY]], i64 0}
// COMMON:  [[INT_TY]] = !{!"int", [[CHAR]], i64 0}
// DEFAULT: [[ANYPTR]] = !{[[ANY_POINTER]],  [[ANY_POINTER]], i64 0}
// DEFAULT: [[P2VOID]] = !{[[P2VOID_TY:!.+]], [[P2VOID_TY]], i64 0}
// DEFAULT: [[P2VOID_TY]] = !{!"p2 void", [[ANY_POINTER]], i64 0}
// DEFAULT: [[P1VOID]] = !{[[P1VOID_TY:!.+]], [[P1VOID_TY]], i64 0}
// DEFAULT: [[P1VOID_TY]] = !{!"p1 void", [[ANY_POINTER]], i64 0}
