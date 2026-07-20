// RUN: %clang_cc1 -triple i386-pc-win32 -emit-llvm -fconvergent-functions -o - < %s | FileCheck -check-prefixes=CHECK,CONVFUNC %s
// RUN: %clang_cc1 -triple i386-pc-win32 -emit-llvm -o - < %s | FileCheck -check-prefixes=CHECK,NOCONVFUNC %s

// Test that the -fconvergent-functions flag works

// CHECK: define {{.*}} @func() #[[ATTR:[0-9]+]]
void func(void) { }

// CONVFUNC: define {{.*}} @nofunc() #[[NOATTR:[0-9]+]]
__attribute__((noconvergent)) void nofunc(void) { }

// CHECK: attributes #[[ATTR]] = {
// NOCONVFUNC-NOT: convergent
// CONVFUNC-SAME: convergent
// CHECK-SAME: }

// CONVFUNC: attributes #[[NOATTR]] = {
// CONVFUNC-NOT: convergent
// CONVFUNC-SAME: }
