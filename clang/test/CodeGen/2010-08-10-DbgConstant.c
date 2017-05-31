// RUN: %clang_cc1 -S -emit-llvm -debug-info-kind=limited  %s -o - | FileCheck %s
// CHECK: !DIGlobalVariable(

static const unsigned int ro = 201;
void bar(int);
void foo() { bar(ro); }
