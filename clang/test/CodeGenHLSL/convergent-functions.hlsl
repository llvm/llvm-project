// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.4-library -emit-llvm -disable-llvm-passes -o - %s | FileCheck -check-prefixes=CHECK,CONVFUNC %s 

// CHECK: attributes
// NOCONVFUNC-NOT: convergent
// CONVFUNC-SAME: convergent
// CHECK-SAME: }
void fn() {
};
