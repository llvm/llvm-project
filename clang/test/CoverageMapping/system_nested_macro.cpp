// Check nested macro handling when including a system header.
// RUN: %clang_cc1 -std=c++11 -isystem %S/Inputs -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -emit-llvm -main-file-name system_nested_macro.cpp -o - %s | FileCheck %s --check-prefixes=CHECK,X_SYS
// RUN: %clang_cc1 -std=c++11 -isystem %S/Inputs -fprofile-instrument=clang -fcoverage-mapping -dump-coverage-mapping -mllvm -system-headers-coverage=true -emit-llvm -main-file-name system_nested_macro.cpp -o - %s | FileCheck %s --check-prefixes=CHECK,W_SYS
//
// CHECK-LABEL: main:
// CHECK: File 0, [[@LINE+6]]:12 -> [[@LINE+9]]:2 = #0
// X_SYS: File 1, [[@LINE+6]]:11 -> [[@LINE+6]]:12 = #0
// X_SYS-NOT: Expansion,
// W_SYS: Expansion,File 0, [[@LINE+5]]:10 -> [[@LINE+5]]:11 = #0 (Expanded file = 1)
// W_SYS: File 1, 1:1 -> 3:1 = #0

int main() {
#define X ;
#include <nested.h>
}
