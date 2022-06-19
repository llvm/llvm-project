// ; Check that the -flto=thin option emits a ThinLTO summary
// RUN: %clang_cc1 -flto=thin -emit-llvm-bc < %s | llvm-bcanalyzer -dump | FileCheck %s
// CHECK: <GLOBALVAL_SUMMARY_BLOCK
//
// ; Check that we do not emit a summary for regular LTO on Apple platforms
// RUN: %clang_cc1 -flto -triple x86_64-apple-darwin -emit-llvm-bc < %s | llvm-bcanalyzer -dump | FileCheck --check-prefix=LTO %s
// LTO-NOT: GLOBALVAL_SUMMARY_BLOCK
//
// ; Check that we emit a summary for regular LTO by default elsewhere
// RUN: %clang_cc1 -flto -triple x86_64-pc-linux-gnu -emit-llvm-bc < %s | llvm-bcanalyzer -dump | FileCheck --check-prefix=LTOINDEX %s
// LTOINDEX: <FULL_LTO_GLOBALVAL_SUMMARY_BLOCK
//
// ; Simulate -save-temps and check that it works (!"ThinLTO" module flag not added multiple times)
// RUN: %clang_cc1 -flto -triple x86_64-pc-linux-gnu -emit-llvm-bc -disable-llvm-passes < %s -o %t.bc
// RUN: %clang_cc1 -flto -triple x86_64-pc-linux-gnu -emit-llvm-bc -x ir < %t.bc | llvm-bcanalyzer -dump | FileCheck --check-prefix=LTOINDEX %s

int main(void) {}
