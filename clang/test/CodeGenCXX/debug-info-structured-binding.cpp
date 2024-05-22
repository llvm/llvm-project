// RUN: %clang_cc1 -emit-llvm -debug-info-kind=standalone -triple %itanium_abi_triple %s -o - | FileCheck %s --implicit-check-not="call void @llvm.dbg.declare"

// CHECK: call void @llvm.dbg.declare(metadata ptr %{{[a-z]+}}, metadata ![[VAR_0:[0-9]+]], metadata !DIExpression())
// CHECK: call void @llvm.dbg.declare(metadata ptr %{{[0-9]+}}, metadata ![[VAR_1:[0-9]+]], metadata !DIExpression())
// CHECK: call void @llvm.dbg.declare(metadata ptr %{{[0-9]+}}, metadata ![[VAR_2:[0-9]+]], metadata !DIExpression(DW_OP_plus_uconst, 4))
// CHECK: call void @llvm.dbg.declare(metadata ptr %{{[0-9]+}}, metadata ![[VAR_3:[0-9]+]], metadata !DIExpression(DW_OP_deref))
// CHECK: call void @llvm.dbg.declare(metadata ptr %{{[0-9]+}}, metadata ![[VAR_4:[0-9]+]], metadata !DIExpression(DW_OP_deref, DW_OP_plus_uconst, 4))
// CHECK: ![[VAR_0]] = !DILocalVariable(name: "a"
// CHECK: ![[VAR_1]] = !DILocalVariable(name: "x1"
// CHECK: ![[VAR_2]] = !DILocalVariable(name: "y1"
// CHECK: ![[VAR_3]] = !DILocalVariable(name: "x2"
// CHECK: ![[VAR_4]] = !DILocalVariable(name: "y2"

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
