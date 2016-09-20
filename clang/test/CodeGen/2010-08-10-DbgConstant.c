// RUN: %clang_cc1 -S -emit-llvm -debug-info-kind=limited  %s -o - | FileCheck %s
// CHECK: !DIGlobalVariable({{.*}}, expr: [[EXPR:![0-9]+]])
// CHECK: [[EXPR]] = !DIExpression(DW_OP_constu, 201, DW_OP_stack_value)

static const unsigned int ro = 201;
void bar(int);
void foo() { bar(ro); }
