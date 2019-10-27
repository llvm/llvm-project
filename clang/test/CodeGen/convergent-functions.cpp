// RUN: %clang_cc1 -triple i386-pc-win32 -emit-llvm -fconvergent-functions -o - < %s | FileCheck -check-prefix=CONVFUNC %s
// RUN: %clang_cc1 -triple i386-pc-win32 -emit-llvm -o - < %s | FileCheck -check-prefix=NOCONVFUNC %s

// Test that the -fconvergent-functions flag works

// CONVFUNC: attributes #0 = { convergent {{.*}} }
// NOCONVFUNC-NOT: convergent
void func() { }
