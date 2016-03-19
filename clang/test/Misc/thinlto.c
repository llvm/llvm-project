// RUN: %clang_cc1 -flto=thin -emit-llvm-bc < %s | llvm-bcanalyzer -dump | FileCheck %s
// CHECK: <GLOBALVAL_SUMMARY_BLOCK
// CHECK-NEXT: <PERMODULE
// CHECK-NEXT: <PERMODULE
// CHECK-NEXT: </GLOBALVAL_SUMMARY_BLOCK

__attribute__((noinline)) void foo() {}

int main() { foo(); }
