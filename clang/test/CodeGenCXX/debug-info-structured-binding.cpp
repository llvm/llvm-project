// RUN: %clang_cc1 -emit-llvm -debug-info-kind=standalone -triple %itanium_abi_triple %s -o - | FileCheck %s --implicit-check-not="call void @llvm.dbg.declare"

// CHECK: #dbg_declare(ptr %{{[a-z]+}}, ![[VAR_0:[0-9]+]], !DIExpression(),
// CHECK: #dbg_declare(ptr %{{[0-9]+}}, ![[VAR_1:[0-9]+]], !DIExpression(),
// CHECK: #dbg_declare(ptr %{{[0-9]+}}, ![[VAR_2:[0-9]+]], !DIExpression(DW_OP_plus_uconst, 4),
// CHECK: #dbg_declare(ptr %{{[0-9]+}}, ![[VAR_3:[0-9]+]], !DIExpression(DW_OP_deref),
// CHECK: #dbg_declare(ptr %{{[0-9]+}}, ![[VAR_4:[0-9]+]], !DIExpression(DW_OP_deref, DW_OP_plus_uconst, 4),
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
