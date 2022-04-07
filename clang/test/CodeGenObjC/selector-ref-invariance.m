// RUN: %clang_cc1 -no-opaque-pointers -triple x86_64-apple-darwin11 -emit-llvm -fblocks -o - %s | FileCheck %s

// rdar://6027699

void test(id x) {
// CHECK: load i8*, i8** @OBJC_SELECTOR_REFERENCES_, align 8, !invariant.load
// CHECK: @objc_msgSend
  [x foo];
}
