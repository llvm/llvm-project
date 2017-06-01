// RUN: %clang_cc1 -flto=thin -emit-llvm-bc < %s | llvm-bcanalyzer -dump | FileCheck %s
// ; Check that the -flto=thin option emits a summary
// CHECK: <GLOBALVAL_SUMMARY_BLOCK
int main() {}
