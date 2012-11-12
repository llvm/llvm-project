// REQUIRES: arm-registered-target
// RUN: %clang_cc1 -triple armv7-apple-darwin9 -target-abi apcs-gnu -emit-llvm -w -o - %s | FileCheck -check-prefix=DARWIN-APCS %s
// RUN: %clang_cc1 -triple armv7-apple-darwin9 -target-abi aapcs  -emit-llvm -w -o - %s | FileCheck -check-prefix=DARWIN-AAPCS %s
// RUN: %clang_cc1 -triple arm-none-linux-gnueabi -target-abi apcs-gnu -emit-llvm -w -o - %s | FileCheck -check-prefix=LINUX-APCS %s
// RUN: %clang_cc1 -triple arm-none-linux-gnueabi -target-abi aapcs  -emit-llvm -w -o - %s | FileCheck -check-prefix=LINUX-AAPCS %s


// DARWIN-APCS: define void @f()
// DARWIN-APCS: call void @g
// DARWIN-AAPCS: define arm_aapcscc void @f()
// DARWIN-AAPCS: call arm_aapcscc void @g
// LINUX-APCS: define arm_apcscc void @f()
// LINUX-APCS: call arm_apcscc void @g
// LINUX-AAPCS: define void @f()
// LINUX-AAPCS: call void @g
void g(void);
void f(void) {
  g();
}
