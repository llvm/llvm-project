// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -x c++ -emit-llvm -fcuda-is-device -debug-info-kind=limited -gheterogeneous-dwarf -o - %s | FileCheck %s
//
// first def for 'a'
// CHECK-LABEL: @_Z1fv()
// CHECK: call void @llvm.dbg.def
//
// one unammed def for the struct holding x1 y1, one def for x1 and one for y1
// CHECK: call void @llvm.dbg.def(metadata ![[X1_LIFETIME:[0-9]+]]
// CHECK: call void @llvm.dbg.def(metadata ![[Y1_LIFETIME:[0-9]+]]
//
// the same for x2 y2, x3 y3 and x4 y4
// CHECK: call void @llvm.dbg.def(metadata ![[X2_LIFETIME:[0-9]+]]
// CHECK: call void @llvm.dbg.def(metadata ![[Y2_LIFETIME:[0-9]+]]
//
// CHECK-LABEL: @_Z1gv()
// CHECK: call void @llvm.dbg.def
//
// CHECK: call void @llvm.dbg.def(metadata ![[X3_LIFETIME:[0-9]+]]
// CHECK: call void @llvm.dbg.def(metadata ![[Y3_LIFETIME:[0-9]+]]
//
// CHECK: call void @llvm.dbg.def(metadata ![[X4_LIFETIME:[0-9]+]]
// CHECK: call void @llvm.dbg.def(metadata ![[Y4_LIFETIME:[0-9]+]]
//
// CHECK: ![[X1_LIFETIME]] = distinct !DILifetime(object: !{{[0-9]+}}, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.A), DIOpConstant(i32 0), DIOpBitOffset(i32)))
// CHECK: ![[Y1_LIFETIME]] = distinct !DILifetime(object: !{{[0-9]+}}, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(%struct.A), DIOpConstant(i32 32), DIOpBitOffset(i32)))
//
// CHECK: ![[X2_LIFETIME]] = distinct !DILifetime(object: !{{[0-9]+}}, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(ptr), DIOpDeref(%struct.A), DIOpConstant(i32 0), DIOpBitOffset(i32)))
// CHECK: ![[Y2_LIFETIME]] = distinct !DILifetime(object: !{{[0-9]+}}, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(ptr), DIOpDeref(%struct.A), DIOpConstant(i32 32), DIOpBitOffset(i32)))
//
// CHECK: ![[X3_LIFETIME]] = distinct !DILifetime(object: !{{[0-9]+}}, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref([2 x i32]), DIOpConstant(i32 0), DIOpByteOffset(i32)))
// CHECK: ![[Y3_LIFETIME]] = distinct !DILifetime(object: !{{[0-9]+}}, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref([2 x i32]), DIOpConstant(i32 4), DIOpByteOffset(i32)))
//
// CHECK: ![[X4_LIFETIME]] = distinct !DILifetime(object: !{{[0-9]+}}, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(ptr), DIOpDeref([2 x i32]), DIOpConstant(i32 0), DIOpByteOffset(i32)))
// CHECK: ![[Y4_LIFETIME]] = distinct !DILifetime(object: !{{[0-9]+}}, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(ptr), DIOpDeref([2 x i32]), DIOpConstant(i32 4), DIOpByteOffset(i32)))

struct A {
  int x;
  int y;
};

int f() {
  A a{10, 20};
  auto [x1, y1] = a;
  auto &[x2, y2] = a;
  return x1 + y1 + x2 + y2;
}

int g() {
  const unsigned A[] = { 10, 20};
  auto [x3, y3] = A;
  auto &[x4, y4] = A;
  return x3 + y3 + x4 + y4;
}
