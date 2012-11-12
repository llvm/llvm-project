// RUN: %clang_cc1 -triple x86_64-apple-darwin11 -emit-llvm -fblocks -o - %s | FileCheck %s

// rdar://6027699

void test(id x) {
// CHECK: load i8** @"\01L_OBJC_SELECTOR_REFERENCES_", !invariant.load
// CHECK: @objc_msgSend
  [x foo];
}
