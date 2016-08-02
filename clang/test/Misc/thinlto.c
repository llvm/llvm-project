// RUN: %clang_cc1 -flto=thin -emit-llvm-bc < %s | llvm-bcanalyzer -dump | FileCheck %s
// CHECK: <FUNCTION_SUMMARY_BLOCK
// CHECK-NEXT: <PERMODULE_ENTRY
// CHECK-NEXT: <PERMODULE_ENTRY
// CHECK-NEXT: </FUNCTION_SUMMARY_BLOCK

__attribute__((noinline)) void foo() {}

int main() { foo(); }
