// RUN: %clang_cc1 -emit-llvm -debug-info-kind=standalone -triple aarch64-arm-none-eabi %s -o - | FileCheck %s

struct S0 {
  unsigned int x : 16;
  unsigned int y : 16;
};

// CHECK-LABEL: define dso_local void @_Z3fS0v
// CHECK:                        alloca %struct.S0, align 4
// CHECK-NEXT:    [[TMP0:%.*]] = alloca %struct.S0, align 4
// CHECK:         call void @llvm.dbg.declare(metadata ptr [[TMP0]], metadata [[S0_A:![0-9]+]], metadata !DIExpression())
// CHECK-NEXT:    call void @llvm.dbg.declare(metadata ptr [[TMP0]], metadata [[S0_B:![0-9]+]], metadata !DIExpression(DW_OP_plus_uconst, 2))
//
void fS0() {
  S0 s0;
  auto [a, b] = s0;
}

struct S1 {
  volatile unsigned int x : 16;
  volatile unsigned int y : 16;
};

// CHECK-LABEL: define dso_local void @_Z3fS1v
// CHECK:                        alloca %struct.S1, align 4
// CHECK-NEXT:    [[TMP0:%.*]] = alloca %struct.S1, align 4
// CHECK:         call void @llvm.dbg.declare(metadata ptr [[TMP0]], metadata [[S1_A:![0-9]+]], metadata !DIExpression())
// CHECK-NEXT:    call void @llvm.dbg.declare(metadata ptr [[TMP0]], metadata [[S1_B:![0-9]+]], metadata !DIExpression(DW_OP_plus_uconst, 2))
//
void fS1() {
  S1 s1;
  auto [a, b] = s1;
}

struct S2 {
  unsigned int x : 8;
  unsigned int y : 8;
};

// CHECK-LABEL: define dso_local void @_Z3fS2v
// CHECK:                        alloca %struct.S2, align 4
// CHECK-NEXT:    [[TMP0:%.*]] = alloca %struct.S2, align 4
// CHECK:         call void @llvm.dbg.declare(metadata ptr [[TMP0]], metadata [[S2_A:![0-9]+]], metadata !DIExpression())
// CHECK-NEXT:    call void @llvm.dbg.declare(metadata ptr [[TMP0]], metadata [[S2_B:![0-9]+]], metadata !DIExpression(DW_OP_plus_uconst, 1))
//
void fS2() {
  S2 s2;
  auto [a, b] = s2;
}

struct S3 {
  volatile unsigned int x : 8;
  volatile unsigned int y : 8;
};

// CHECK-LABEL: define dso_local void @_Z3fS3v
// CHECK:                        alloca %struct.S3, align 4
// CHECK-NEXT:    [[TMP0:%.*]] = alloca %struct.S3, align 4
// CHECK:         call void @llvm.dbg.declare(metadata ptr [[TMP0]], metadata [[S3_A:![0-9]+]], metadata !DIExpression())
// CHECK-NEXT:    call void @llvm.dbg.declare(metadata ptr [[TMP0]], metadata [[S3_B:![0-9]+]], metadata !DIExpression(DW_OP_plus_uconst, 1))
//
void fS3() {
  S3 s3;
  auto [a, b] = s3;
}

struct S4 {
  unsigned int x : 8;
  unsigned int y : 16;
};

// CHECK-LABEL: define dso_local void @_Z3fS4v
// CHECK:                        alloca %struct.S4, align 4
// CHECK-NEXT:    [[TMP0:%.*]] = alloca %struct.S4, align 4
// CHECK:         call void @llvm.dbg.declare(metadata ptr [[TMP0]], metadata [[S4_A:![0-9]+]], metadata !DIExpression())
// CHECK-NEXT:    call void @llvm.dbg.declare(metadata ptr [[TMP0]], metadata [[S4_B:![0-9]+]], metadata !DIExpression(DW_OP_plus_uconst, 1))
//
void fS4() {
  S4 s4;
  auto [a, b] = s4;
}

struct S5 {
  volatile unsigned int x : 8;
  volatile unsigned int y : 16;
};

// CHECK-LABEL: define dso_local void @_Z3fS5v
// CHECK:                        alloca %struct.S5, align 4
// CHECK-NEXT:    [[TMP0:%.*]] = alloca %struct.S5, align 4
// CHECK:         call void @llvm.dbg.declare(metadata ptr [[TMP0]], metadata [[S5_A:![0-9]+]], metadata !DIExpression())
// CHECK-NEXT:    call void @llvm.dbg.declare(metadata ptr [[TMP0]], metadata [[S5_B:![0-9]+]], metadata !DIExpression(DW_OP_plus_uconst, 1))
//
void fS5() {
  S5 s5;
  auto [a, b] = s5;
}

struct S6 {
  unsigned int x : 16;
  unsigned int y : 8;
};

// CHECK-LABEL: define dso_local void @_Z3fS6v
// CHECK:                        alloca %struct.S6, align 4
// CHECK-NEXT:    [[TMP0:%.*]] = alloca %struct.S6, align 4
// CHECK:         call void @llvm.dbg.declare(metadata ptr [[TMP0]], metadata [[S6_A:![0-9]+]], metadata !DIExpression())
// CHECK-NEXT:    call void @llvm.dbg.declare(metadata ptr [[TMP0]], metadata [[S6_B:![0-9]+]], metadata !DIExpression(DW_OP_plus_uconst, 2))
//
void fS6() {
  S6 s6;
  auto [a, b] = s6;
}

struct S7 {
  volatile unsigned int x : 16;
  volatile unsigned int y : 8;
};

// CHECK-LABEL: define dso_local void @_Z3fS7v
// CHECK:                        alloca %struct.S7, align 4
// CHECK-NEXT:    [[TMP0:%.*]] = alloca %struct.S7, align 4
// CHECK:         call void @llvm.dbg.declare(metadata ptr [[TMP0]], metadata [[S7_A:![0-9]+]], metadata !DIExpression())
// CHECK-NEXT:    call void @llvm.dbg.declare(metadata ptr [[TMP0]], metadata [[S7_B:![0-9]+]], metadata !DIExpression(DW_OP_plus_uconst, 2))
//
void fS7() {
  S7 s7;
  auto [a, b] = s7;
}

struct S8 {
  unsigned int x : 16;
  volatile unsigned int y : 16;
};

// CHECK-LABEL: define dso_local void @_Z3fS8v
// CHECK:                        alloca %struct.S8, align 4
// CHECK-NEXT:    [[TMP0:%.*]] = alloca %struct.S8, align 4
// CHECK:         call void @llvm.dbg.declare(metadata ptr [[TMP0]], metadata [[S8_A:![0-9]+]], metadata !DIExpression())
// CHECK-NEXT:    call void @llvm.dbg.declare(metadata ptr [[TMP0]], metadata [[S8_B:![0-9]+]], metadata !DIExpression(DW_OP_plus_uconst, 2))
//
void fS8() {
  S8 s8;
  auto [a, b] = s8;
}

struct S9 {
  unsigned int x : 16;
  unsigned int y : 32;
};

// CHECK-LABEL: define dso_local void @_Z3fS9v
// CHECK:                        alloca %struct.S9, align 4
// CHECK-NEXT:    [[TMP0:%.*]] = alloca %struct.S9, align 4
// CHECK:         call void @llvm.dbg.declare(metadata ptr [[TMP0]], metadata [[S9_A:![0-9]+]], metadata !DIExpression())
// CHECK-NEXT:    call void @llvm.dbg.declare(metadata ptr [[TMP0]], metadata [[S9_B:![0-9]+]], metadata !DIExpression(DW_OP_plus_uconst, 4))
//
void fS9() {
  S9 s9;
  auto [a, b] = s9;
}

struct S10 {
  const unsigned int x : 8;
  const volatile unsigned int y : 8;

// CHECK-LABEL: define dso_local void @_Z4fS10v
// CHECK:                        alloca %struct.S10, align 4
// CHECK-NEXT:    [[TMP0:%.*]] = alloca %struct.S10, align 4
// CHECK:         call void @llvm.dbg.declare(metadata ptr [[TMP0]], metadata [[S10_A:![0-9]+]], metadata !DIExpression())
// CHECK-NEXT:    call void @llvm.dbg.declare(metadata ptr [[TMP0]], metadata [[S10_B:![0-9]+]], metadata !DIExpression(DW_OP_plus_uconst, 1))
//
  S10() : x(0), y(0) {}
};

void fS10() {
  S10 s10;
  auto [a, b] = s10;
}

// It's currently not possible to produce complete debug information for the following cases.
// Confirm that no wrong debug info is output.
// Once this is implemented, these tests should be amended.
struct S11 {
  unsigned int x : 15;
  unsigned int y : 16;
};

// CHECK-LABEL: define dso_local void @_Z4fS11v
// CHECK:                        alloca %struct.S11, align 4
// CHECK-NEXT:    [[TMP0:%.*]] = alloca %struct.S11, align 4
// CHECK-NOT:     call void @llvm.dbg.declare(metadata ptr [[TMP0]]
//
void fS11() {
  S11 s11;
  auto [a, b] = s11;
}

struct S12 {
  unsigned int x : 16;
  unsigned int y : 17;
};

// CHECK-LABEL: define dso_local void @_Z4fS12v
// CHECK:                        alloca %struct.S12, align 4
// CHECK-NEXT:    [[TMP0:%.*]] = alloca %struct.S12, align 4
// CHECK:         call void @llvm.dbg.declare(metadata ptr [[TMP0]], metadata [[S12_A:![0-9]+]], metadata !DIExpression())
// CHECK-NOT:     call void @llvm.dbg.declare(metadata ptr [[TMP0]]
//
void fS12() {
  S12 s12;
  auto [a, b] = s12;
}

struct __attribute__((packed)) S13 {
  unsigned int x : 15;
  unsigned int y : 16;
};

// CHECK-LABEL: define dso_local void @_Z4fS13v
// CHECK:                        alloca %struct.S13, align 1
// CHECK-NEXT:    [[TMP0:%.*]] = alloca %struct.S13, align 1
// CHECK-NOT:     call void @llvm.dbg.declare(metadata ptr [[TMP0]]
//
void fS13() {
  S13 s13;
  auto [a, b] = s13;
}

// CHECK: [[UINT_TY:![0-9]+]] = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
// CHECK: [[S0_A]] = !DILocalVariable(name: "a", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[USHORT_TY:![0-9]+]])
// CHECK: [[USHORT_TY]] = !DIBasicType(name: "unsigned short", size: 16, encoding: DW_ATE_unsigned)
// CHECK: [[S0_B]] = !DILocalVariable(name: "b", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[USHORT_TY]])

// CHECK: [[VOLATILE_UINT_TY:![0-9]+]] = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: [[UINT_TY]])
// CHECK: [[S1_A]] = !DILocalVariable(name: "a", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[VOLATILE_USHORT_TY:![0-9]+]])
// CHECK: [[VOLATILE_USHORT_TY]] = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: [[USHORT_TY]])
// CHECK: [[S1_B]] = !DILocalVariable(name: "b", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[VOLATILE_USHORT_TY]])

// CHECK: [[S2_A]] = !DILocalVariable(name: "a", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[UCHAR_TY:![0-9]+]])
// CHECK: [[UCHAR_TY]] = !DIBasicType(name: "unsigned char", size: 8, encoding: DW_ATE_unsigned_char)
// CHECK: [[S2_B]] = !DILocalVariable(name: "b", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[UCHAR_TY]])

// CHECK: [[S3_A]] = !DILocalVariable(name: "a", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[VOLATILE_UCHAR_TY:![0-9]+]])
// CHECK: [[VOLATILE_UCHAR_TY]] = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: [[UCHAR_TY]])
// CHECK: [[S3_B]] = !DILocalVariable(name: "b", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[VOLATILE_UCHAR_TY]])

// CHECK: [[S4_A]] = !DILocalVariable(name: "a", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[UCHAR_TY]])
// CHECK: [[S4_B]] = !DILocalVariable(name: "b", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[USHORT_TY]])

// CHECK: [[S5_A]] = !DILocalVariable(name: "a", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[VOLATILE_UCHAR_TY]])
// CHECK: [[S5_B]] = !DILocalVariable(name: "b", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[VOLATILE_USHORT_TY]])

// CHECK: [[S6_A]] = !DILocalVariable(name: "a", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[USHORT_TY]])
// CHECK: [[S6_B]] = !DILocalVariable(name: "b", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[UCHAR_TY]])

// CHECK: [[S7_A]] = !DILocalVariable(name: "a", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[VOLATILE_USHORT_TY]])
// CHECK: [[S7_B]] = !DILocalVariable(name: "b", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[VOLATILE_UCHAR_TY]])

// CHECK: [[S8_A]] = !DILocalVariable(name: "a", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[USHORT_TY]])
// CHECK: [[S8_B]] = !DILocalVariable(name: "b", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[VOLATILE_USHORT_TY]])

// CHECK: [[S9_A]] = !DILocalVariable(name: "a", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[USHORT_TY]])
// CHECK: [[S9_B]] = !DILocalVariable(name: "b", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[UINT_TY]])

// CHECK: [[CONST_UINT_TY:![0-9]+]] = !DIDerivedType(tag: DW_TAG_const_type, baseType: [[UINT_TY]])
// CHECK: [[S10_A]] = !DILocalVariable(name: "a", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[CONST_UCHAR_TY:![0-9]+]])
// CHECK: [[CONST_UCHAR_TY]] = !DIDerivedType(tag: DW_TAG_const_type, baseType: [[UCHAR_TY]])
// CHECK: [[S10_B]] = !DILocalVariable(name: "b", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[CONST_VOLATILE_UCHAR_TY:![0-9]+]])
// CHECK: [[CONST_VOLATILE_UCHAR_TY]] = !DIDerivedType(tag: DW_TAG_const_type, baseType: [[VOLATILE_UCHAR_TY]])

// S11
// CHECK-NOT: !DILocalVariable(name: "a"
// CHECK-NOT: !DILocalVariable(name: "b"

// S12
// CHECK: [[S12_A]] = !DILocalVariable(name: "a", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[USHORT_TY]])
// CHECK-NOT: !DILocalVariable(name: "b"

// S13
// CHECK-NOT: !DILocalVariable(name: "a"
// CHECK-NOT: !DILocalVariable(name: "b"