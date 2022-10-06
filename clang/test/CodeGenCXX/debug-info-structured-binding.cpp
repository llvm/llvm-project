// RUN: %clang_cc1 -emit-llvm -debug-info-kind=standalone -triple %itanium_abi_triple %s -o - | FileCheck %s

// CHECK: call void @llvm.dbg.declare(metadata ptr %{{[0-9]+}}, metadata !{{[0-9]+}}, metadata !DIExpression())
// CHECK: call void @llvm.dbg.declare(metadata ptr %{{[0-9]+}}, metadata !{{[0-9]+}}, metadata !DIExpression(DW_OP_plus_uconst, {{[0-9]+}}))
// CHECK: call void @llvm.dbg.declare(metadata ptr %{{[0-9]+}}, metadata !{{[0-9]+}}, metadata !DIExpression())
// CHECK: call void @llvm.dbg.declare(metadata ptr %{{[0-9]+}}, metadata !{{[0-9]+}}, metadata !DIExpression(DW_OP_deref))
// CHECK: call void @llvm.dbg.declare(metadata ptr %{{[0-9]+}}, metadata !{{[0-9]+}}, metadata !DIExpression(DW_OP_deref, DW_OP_plus_uconst, {{[0-9]+}}))
// CHECK: call void @llvm.dbg.declare(metadata ptr %{{[0-9]+}}, metadata !{{[0-9]+}}, metadata !DIExpression())
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
