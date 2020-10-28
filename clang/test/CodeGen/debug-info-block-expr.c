// RUN: %clang_cc1 -fblocks -debug-info-kind=limited -emit-llvm -o - %s | FileCheck %s
// Verify that the desired DIExpression are generated for blocks.

void test() {
// CHECK: call void @llvm.dbg.declare({{.*}}!DIExpression(DW_OP_plus_uconst, {{[0-9]+}}, DW_OP_deref, DW_OP_plus_uconst, {{[0-9]+}}){{.*}})
  __block int i;
// CHECK: call void @llvm.dbg.declare({{.*}}!DIExpression(DW_OP_deref, DW_OP_plus_uconst, {{[0-9]+}}, DW_OP_deref, DW_OP_plus_uconst, {{[0-9]+}}, DW_OP_deref, DW_OP_plus_uconst, {{[0-9]+}}){{.*}})
  ^ { i = 1; }();
}
