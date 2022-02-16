// RUN: %clang_cc1 -triple i386-pc-win32 -emit-llvm -fconvergent-functions -o - < %s | FileCheck -check-prefixes=CHECK,CONVFUNC %s
// RUN: %clang_cc1 -triple i386-pc-win32 -emit-llvm -o - < %s | FileCheck -check-prefixes=CHECK,NOCONVFUNC %s

// Test that the -fconvergent-functions flag works

// CHECK: attributes #0 = {
// NOCONVFUNC-NOT: convergent
// CONVFUNC-SAME: convergent
// CHECK-SAME: }
void func(void) { }
