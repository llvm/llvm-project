// RUN: %clang_cc1 -triple s390x-linux-gnu -emit-llvm -o - -ftrapv \
// RUN:     -ftrapv-handler myhandler %s | FileCheck %s
//
// Check that the two i8 arguments of the -ftrapv-handler function (operation
// code and operand bit-width) carry zeroext.
//
// CHECK: declare i64 @myhandler(i64, i64, i8 zeroext, i8 zeroext, ...)

void test(int a, int b) {
  (void)(a + b);
}
