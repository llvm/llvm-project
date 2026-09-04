// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited  %s -o - | FileCheck %s
// CHECK: !DIGlobalVariableExpression(var: [[VAR:.*]], expr: !DIExpression(DW_OP_constu, 201, DW_OP_stack_value))

static const unsigned int ro = 201;
void bar(int);
void foo(void) { bar(ro); }
