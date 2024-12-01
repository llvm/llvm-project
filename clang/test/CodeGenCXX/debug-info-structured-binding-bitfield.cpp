// RUN: %clang_cc1 -emit-llvm -debug-info-kind=standalone -triple aarch64 %s -o - | FileCheck %s

struct S0 {
  unsigned int x : 16;
  unsigned int y : 16;
};

// CHECK-LABEL: define dso_local void @_Z3fS0v
// CHECK:                        alloca %struct.S0, align 4
// CHECK-NEXT:    [[TMP0:%.*]] = alloca %struct.S0, align 4
// CHECK:         #dbg_declare(ptr [[TMP0]], [[S0_A:![0-9]+]], !DIExpression(DW_OP_LLVM_extract_bits_zext, 0, 16),
// CHECK-NEXT:    #dbg_declare(ptr [[TMP0]], [[S0_B:![0-9]+]], !DIExpression(DW_OP_LLVM_extract_bits_zext, 16, 16),
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
// CHECK:         #dbg_declare(ptr [[TMP0]], [[S1_A:![0-9]+]], !DIExpression(DW_OP_LLVM_extract_bits_zext, 0, 16),
// CHECK-NEXT:    #dbg_declare(ptr [[TMP0]], [[S1_B:![0-9]+]], !DIExpression(DW_OP_LLVM_extract_bits_zext, 16, 16),
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
// CHECK:         #dbg_declare(ptr [[TMP0]], [[S2_A:![0-9]+]], !DIExpression(DW_OP_LLVM_extract_bits_zext, 0, 8),
// CHECK-NEXT:    #dbg_declare(ptr [[TMP0]], [[S2_B:![0-9]+]], !DIExpression(DW_OP_LLVM_extract_bits_zext, 8, 8),
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
// CHECK:         #dbg_declare(ptr [[TMP0]], [[S3_A:![0-9]+]], !DIExpression(DW_OP_LLVM_extract_bits_zext, 0, 8),
// CHECK-NEXT:    #dbg_declare(ptr [[TMP0]], [[S3_B:![0-9]+]], !DIExpression(DW_OP_LLVM_extract_bits_zext, 8, 8),
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
// CHECK:         #dbg_declare(ptr [[TMP0]], [[S4_A:![0-9]+]], !DIExpression(DW_OP_LLVM_extract_bits_zext, 0, 8),
// CHECK-NEXT:    #dbg_declare(ptr [[TMP0]], [[S4_B:![0-9]+]], !DIExpression(DW_OP_LLVM_extract_bits_zext, 8, 16),
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
// CHECK:         #dbg_declare(ptr [[TMP0]], [[S5_A:![0-9]+]], !DIExpression(DW_OP_LLVM_extract_bits_zext, 0, 8),
// CHECK-NEXT:    #dbg_declare(ptr [[TMP0]], [[S5_B:![0-9]+]], !DIExpression(DW_OP_LLVM_extract_bits_zext, 8, 16),
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
// CHECK:         #dbg_declare(ptr [[TMP0]], [[S6_A:![0-9]+]], !DIExpression(DW_OP_LLVM_extract_bits_zext, 0, 16),
// CHECK-NEXT:    #dbg_declare(ptr [[TMP0]], [[S6_B:![0-9]+]], !DIExpression(DW_OP_LLVM_extract_bits_zext, 16, 8),
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
// CHECK:         #dbg_declare(ptr [[TMP0]], [[S7_A:![0-9]+]], !DIExpression(DW_OP_LLVM_extract_bits_zext, 0, 16),
// CHECK-NEXT:    #dbg_declare(ptr [[TMP0]], [[S7_B:![0-9]+]], !DIExpression(DW_OP_LLVM_extract_bits_zext, 16, 8),
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
// CHECK:         #dbg_declare(ptr [[TMP0]], [[S8_A:![0-9]+]], !DIExpression(DW_OP_LLVM_extract_bits_zext, 0, 16),
// CHECK-NEXT:    #dbg_declare(ptr [[TMP0]], [[S8_B:![0-9]+]], !DIExpression(DW_OP_LLVM_extract_bits_zext, 16, 16),
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
// CHECK:         #dbg_declare(ptr [[TMP0]], [[S9_A:![0-9]+]], !DIExpression(DW_OP_LLVM_extract_bits_zext, 0, 16),
// CHECK-NEXT:    #dbg_declare(ptr [[TMP0]], [[S9_B:![0-9]+]], !DIExpression(DW_OP_LLVM_extract_bits_zext, 32, 32),
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
// CHECK:         #dbg_declare(ptr [[TMP0]], [[S10_A:![0-9]+]], !DIExpression(DW_OP_LLVM_extract_bits_zext, 0, 8),
// CHECK-NEXT:    #dbg_declare(ptr [[TMP0]], [[S10_B:![0-9]+]], !DIExpression(DW_OP_LLVM_extract_bits_zext, 8, 8),
//
  S10() : x(0), y(0) {}
};

void fS10() {
  S10 s10;
  auto [a, b] = s10;
}

struct S11 {
  unsigned int x : 15;
  unsigned int y : 16;
};

// CHECK-LABEL: define dso_local void @_Z4fS11v
// CHECK:                        alloca %struct.S11, align 4
// CHECK-NEXT:    [[TMP0:%.*]] = alloca %struct.S11, align 4
// CHECK:         #dbg_declare(ptr [[TMP0]], [[S11_A:![0-9]+]], !DIExpression(DW_OP_LLVM_extract_bits_zext, 0, 15),
// CHECK-NEXT:    #dbg_declare(ptr [[TMP0]], [[S11_B:![0-9]+]], !DIExpression(DW_OP_LLVM_extract_bits_zext, 15, 16),
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
// CHECK:         #dbg_declare(ptr [[TMP0]], [[S12_A:![0-9]+]], !DIExpression(DW_OP_LLVM_extract_bits_zext, 0, 16),
// CHECK-NEXT:    #dbg_declare(ptr [[TMP0]], [[S12_B:![0-9]+]], !DIExpression(DW_OP_LLVM_extract_bits_zext, 32, 17),
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
// CHECK:         #dbg_declare(ptr [[TMP0]], [[S13_A:![0-9]+]], !DIExpression(DW_OP_LLVM_extract_bits_zext, 0, 15),
// CHECK-NEXT:    #dbg_declare(ptr [[TMP0]], [[S13_B:![0-9]+]], !DIExpression(DW_OP_LLVM_extract_bits_zext, 15, 16),
//
void fS13() {
  S13 s13;
  auto [a, b] = s13;
}

struct S14 {
  signed int x;
  signed int y : 7;
};

// CHECK-LABEL: define dso_local void @_Z4fS14v
// CHECK:                        alloca %struct.S14, align 4
// CHECK-NEXT:    [[TMP0:%.*]] = alloca %struct.S14, align 4
// CHECK:         #dbg_declare(ptr [[TMP0]], [[S14_A:![0-9]+]], !DIExpression(),
// CHECK-NEXT:    #dbg_declare(ptr [[TMP0]], [[S14_B:![0-9]+]], !DIExpression(DW_OP_plus_uconst, 4, DW_OP_LLVM_extract_bits_sext, 0, 7),
//
void fS14() {
  S14 s14;
  auto [a, b] = s14;
}

struct S15 {
  signed int x : 123;
  unsigned int y : 987;
};

// CHECK-LABEL: define dso_local void @_Z4fS15v
// CHECK:                        alloca %struct.S15, align 8
// CHECK-NEXT:    [[TMP0:%.*]] = alloca %struct.S15, align 8
// CHECK:         #dbg_declare(ptr [[TMP0]], [[S15_A:![0-9]+]], !DIExpression(DW_OP_LLVM_extract_bits_sext, 0, 32),
// CHECK-NEXT:    #dbg_declare(ptr [[TMP0]], [[S15_B:![0-9]+]], !DIExpression(DW_OP_plus_uconst, 16, DW_OP_LLVM_extract_bits_zext, 0, 32),
//
void fS15() {
  S15 s15;
  auto [a, b] = s15;
}

// CHECK: [[UINT_TY:![0-9]+]] = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
// CHECK: [[S0_A]] = !DILocalVariable(name: "a", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[UINT_TY]])
// CHECK: [[S0_B]] = !DILocalVariable(name: "b", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[UINT_TY]])

// CHECK: [[VOLATILE_UINT_TY:![0-9]+]] = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: [[UINT_TY]])
// CHECK: [[S1_A]] = !DILocalVariable(name: "a", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[VOLATILE_UINT_TY:![0-9]+]])
// CHECK: [[S1_B]] = !DILocalVariable(name: "b", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[VOLATILE_UINT_TY]])

// CHECK: [[S2_A]] = !DILocalVariable(name: "a", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[UINT_TY]])
// CHECK: [[S2_B]] = !DILocalVariable(name: "b", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[UINT_TY]])

// CHECK: [[S3_A]] = !DILocalVariable(name: "a", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[VOLATILE_UINT_TY]])
// CHECK: [[S3_B]] = !DILocalVariable(name: "b", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[VOLATILE_UINT_TY]])

// CHECK: [[S4_A]] = !DILocalVariable(name: "a", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[UINT_TY]])
// CHECK: [[S4_B]] = !DILocalVariable(name: "b", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[UINT_TY]])

// CHECK: [[S5_A]] = !DILocalVariable(name: "a", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[VOLATILE_UINT_TY]])
// CHECK: [[S5_B]] = !DILocalVariable(name: "b", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[VOLATILE_UINT_TY]])

// CHECK: [[S6_A]] = !DILocalVariable(name: "a", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[UINT_TY]])
// CHECK: [[S6_B]] = !DILocalVariable(name: "b", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[UINT_TY]])

// CHECK: [[S7_A]] = !DILocalVariable(name: "a", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[VOLATILE_UINT_TY]])
// CHECK: [[S7_B]] = !DILocalVariable(name: "b", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[VOLATILE_UINT_TY]])

// CHECK: [[S8_A]] = !DILocalVariable(name: "a", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[UINT_TY]])
// CHECK: [[S8_B]] = !DILocalVariable(name: "b", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[VOLATILE_UINT_TY]])

// CHECK: [[S9_A]] = !DILocalVariable(name: "a", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[UINT_TY]])
// CHECK: [[S9_B]] = !DILocalVariable(name: "b", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[UINT_TY]])

// CHECK: [[CONST_UINT_TY:![0-9]+]] = !DIDerivedType(tag: DW_TAG_const_type, baseType: [[UINT_TY]])
// CHECK: [[CONST_VOLATILE_UINT_TY:![0-9]+]] = !DIDerivedType(tag: DW_TAG_const_type, baseType: [[VOLATILE_UINT_TY]])
// CHECK: [[S10_A]] = !DILocalVariable(name: "a", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[CONST_UINT_TY]])
// CHECK: [[S10_B]] = !DILocalVariable(name: "b", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[CONST_VOLATILE_UINT_TY]])

// S11
// CHECK: [[S11_A]] = !DILocalVariable(name: "a", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[UINT_TY]])
// CHECK: [[S11_B]] = !DILocalVariable(name: "b", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[UINT_TY]])

// S12
// CHECK: [[S12_A]] = !DILocalVariable(name: "a", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[UINT_TY]])
// CHECK: [[S12_B]] = !DILocalVariable(name: "b", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[UINT_TY]])

// S13
// CHECK: [[S13_A]] = !DILocalVariable(name: "a", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[UINT_TY]])
// CHECK: [[S13_B]] = !DILocalVariable(name: "b", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[UINT_TY]])

// S14
// CHECK: [[SINT_TY:![0-9]+]] = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK: [[S14_A]] = !DILocalVariable(name: "a", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[SINT_TY]])
// CHECK: [[S14_B]] = !DILocalVariable(name: "b", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[SINT_TY]])

// S15
// CHECK: [[S15_A]] = !DILocalVariable(name: "a", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[SINT_TY]])
// CHECK: [[S15_B]] = !DILocalVariable(name: "b", scope: {{.*}}, file: {{.*}}, line: {{.*}}, type: [[UINT_TY]])
