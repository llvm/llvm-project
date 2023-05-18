// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s

// CHECK: @_Z4bad1P8NSObjectP13objc_selectorP11objc_objectS4_
void bad1(struct NSObject *, SEL, id, id) {}
