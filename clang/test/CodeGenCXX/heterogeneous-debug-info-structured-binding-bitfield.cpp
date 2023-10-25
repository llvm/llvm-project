// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x c++ -emit-llvm -fcuda-is-device -debug-info-kind=limited -gheterogeneous-dwarf -o - %s | FileCheck %s

struct S0 {
  unsigned int x : 16;
  unsigned int y : 16;
};

// CHECK-LABEL: define dso_local void @_Z3fS0v
// CHECK: call void @llvm.dbg.def(metadata ![[S0_LIFETIME:[0-9]+]]
// CHECK: call void @llvm.dbg.def(metadata ![[A_S0_LIFETIME:[0-9]+]]
// CHECK: call void @llvm.dbg.def(metadata ![[B_S0_LIFETIME:[0-9]+]]
void fS0() {
  S0 s0;
  auto [a, b] = s0;
}

struct S1 {
  unsigned int x : 8;
  unsigned int y : 8;
};

// CHECK-LABEL: define dso_local void @_Z3fS1v
// CHECK: call void @llvm.dbg.def(metadata ![[S1_LIFETIME:[0-9]+]]
// CHECK: call void @llvm.dbg.def(metadata ![[A_S1_LIFETIME:[0-9]+]]
// CHECK: call void @llvm.dbg.def(metadata ![[B_S1_LIFETIME:[0-9]+]]
void fS1() {
  S1 s1;
  auto [a, b] = s1;
}

struct S2 {
  unsigned int x : 8;
  unsigned int y : 16;
};

// CHECK-LABEL: define dso_local void @_Z3fS2v
// CHECK: call void @llvm.dbg.def(metadata ![[S2_LIFETIME:[0-9]+]]
// CHECK: call void @llvm.dbg.def(metadata ![[A_S2_LIFETIME:[0-9]+]]
// CHECK: call void @llvm.dbg.def(metadata ![[B_S2_LIFETIME:[0-9]+]]
void fS2() {
  S2 s2;
  auto [a, b] = s2;
}

struct S3 {
  unsigned int x : 16;
  unsigned int y : 32;
};

// CHECK-LABEL: define dso_local void @_Z3fS3v
// CHECK: call void @llvm.dbg.def(metadata ![[S3_LIFETIME:[0-9]+]]
// CHECK: call void @llvm.dbg.def(metadata ![[A_S3_LIFETIME:[0-9]+]]
// CHECK: call void @llvm.dbg.def(metadata ![[B_S3_LIFETIME:[0-9]+]]
void fS3() {
  S3 s3;
  auto [a, b] = s3;
}

struct S4 {
  unsigned int x : 16;
  unsigned : 0;
  unsigned int y : 16;
};

// CHECK-LABEL: define dso_local void @_Z3fS4v
// CHECK: call void @llvm.dbg.def(metadata ![[S4_LIFETIME:[0-9]+]]
// CHECK: call void @llvm.dbg.def(metadata ![[A_S4_LIFETIME:[0-9]+]]
// CHECK: call void @llvm.dbg.def(metadata ![[B_S4_LIFETIME:[0-9]+]]
void fS4() {
  S4 s4;
  auto [a, b] = s4;
}

// It's currently not possible to produce complete debug information for the following cases.
// Confirm that no wrong debug info is output.
// Once this is implemented, these tests should be amended.
struct S5 {
  unsigned int x : 15;
  unsigned int y : 16;
};

// CHECK-LABEL: define dso_local void @_Z3fS5v
// CHECK: call void @llvm.dbg.def(metadata ![[S5_LIFETIME:[0-9]+]]
void fS5() {
  S5 s5;
  auto [a, b] = s5;
}

// Currently, LLVM when it emits the structured binding for a bitfield it also emits the DIExpression as an i32 (which mismaches the bitfield width)

// CHECK: ![[S0_LIFETIME]] = distinct !DILifetime(object: !{{[0-9]+}}, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.S0)))
// CHECK: ![[A_S0_LIFETIME]] = distinct !DILifetime(object: !{{[0-9]+}}, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.S0), DIOpConstant(i32 0), DIOpBitOffset(i32)))
// CHECK: ![[B_S0_LIFETIME]] = distinct !DILifetime(object: !{{[0-9]+}}, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.S0), DIOpConstant(i32 16), DIOpBitOffset(i32)))

// CHECK: ![[S1_LIFETIME]] = distinct !DILifetime(object: !{{[0-9]+}}, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.S1)))
// CHECK: ![[A_S1_LIFETIME]] = distinct !DILifetime(object: !{{[0-9]+}}, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.S1), DIOpConstant(i32 0), DIOpBitOffset(i32)))
// CHECK: ![[B_S1_LIFETIME]] = distinct !DILifetime(object: !{{[0-9]+}}, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.S1), DIOpConstant(i32 8), DIOpBitOffset(i32)))

// CHECK: ![[S2_LIFETIME]] = distinct !DILifetime(object: !{{[0-9]+}}, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.S2)))
// CHECK: ![[A_S2_LIFETIME]] = distinct !DILifetime(object: !{{[0-9]+}}, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.S2), DIOpConstant(i32 0), DIOpBitOffset(i32)))
// CHECK: ![[B_S2_LIFETIME]] = distinct !DILifetime(object: !{{[0-9]+}}, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.S2), DIOpConstant(i32 8), DIOpBitOffset(i32)))

// CHECK: ![[S3_LIFETIME]] = distinct !DILifetime(object: !{{[0-9]+}}, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.S3)))
// CHECK: ![[A_S3_LIFETIME]] = distinct !DILifetime(object: !{{[0-9]+}}, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.S3), DIOpConstant(i32 0), DIOpBitOffset(i32)))
// CHECK: ![[B_S3_LIFETIME]] = distinct !DILifetime(object: !{{[0-9]+}}, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.S3), DIOpConstant(i32 32), DIOpBitOffset(i32)))

// CHECK: ![[S4_LIFETIME]] = distinct !DILifetime(object: !{{[0-9]+}}, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.S4)))
// CHECK: ![[A_S4_LIFETIME]] = distinct !DILifetime(object: !{{[0-9]+}}, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.S4), DIOpConstant(i32 0), DIOpBitOffset(i32)))
// CHECK: ![[B_S4_LIFETIME]] = distinct !DILifetime(object: !{{[0-9]+}}, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.S4), DIOpConstant(i32 32), DIOpBitOffset(i32)))

// CHECK: ![[S5_LIFETIME]] = distinct !DILifetime(object: !{{[0-9]+}}, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.S5)))
