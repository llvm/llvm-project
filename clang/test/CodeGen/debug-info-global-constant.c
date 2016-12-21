// RUN: %clang_cc1 -emit-llvm -debug-info-kind=standalone \
// RUN:      -triple %itanium_abi_triple %s -o - | FileCheck %s

// Debug info for a global constant whose address is taken should be emitted
// exactly once.

// CHECK: @i = internal constant i32 1, align 4, !dbg ![[I:[0-9]+]]
// CHECK: ![[I]] = !DIGlobalVariableExpression(var: ![[VAR:.*]], expr: ![[EXPR:[0-9]+]])
// CHECK: ![[VAR]] = distinct !DIGlobalVariable(name: "i",
// CHECK: !DICompileUnit({{.*}}globals: ![[GLOBALS:[0-9]+]])
// CHECK: ![[GLOBALS]] = !{![[I]]}
// CHECK: ![[EXPR]] = !DIExpression(DW_OP_constu, 1, DW_OP_stack_value)
static const int i = 1;

void g(const int *, int);
void f() {
  g(&i, i);
}
