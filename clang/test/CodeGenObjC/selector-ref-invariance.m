// RUN: %clang_cc1 -triple x86_64-apple-darwin11 -emit-llvm -fblocks -o - %s | FileCheck %s

void test(id x) {
// CHECK: load ptr, ptr @OBJC_SELECTOR_REFERENCES_, align 8, !invariant.load
// CHECK: @objc_msgSend
  [x foo];
}
