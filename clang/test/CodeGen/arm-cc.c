// REQUIRES: arm-registered-target
// RUN: %clang_cc1 -triple armv7-apple-darwin9 -target-abi apcs-gnu -emit-llvm -w -o - %s | FileCheck -check-prefix=DARWIN-APCS %s
// RUN: %clang_cc1 -triple armv7-apple-darwin9 -target-abi aapcs  -emit-llvm -w -o - %s | FileCheck -check-prefix=DARWIN-AAPCS %s
// RUN: %clang_cc1 -triple arm-none-linux-gnueabi -target-abi apcs-gnu -emit-llvm -w -o - %s | FileCheck -check-prefix=LINUX-APCS %s
// RUN: %clang_cc1 -triple arm-none-linux-gnueabi -target-abi aapcs  -emit-llvm -w -o - %s | FileCheck -check-prefix=LINUX-AAPCS %s
// RUN: %clang_cc1 -triple arm-none-linux-musleabi -target-abi apcs-gnu -emit-llvm -w -o - %s | FileCheck -check-prefix=LINUX-APCS %s
// RUN: %clang_cc1 -triple arm-none-linux-musleabi -target-abi aapcs  -emit-llvm -w -o - %s | FileCheck -check-prefix=LINUX-AAPCS %s
// RUN: %clang_cc1 -triple armv7-none-eabihf -target-abi aapcs-vfp -emit-llvm -w -o - %s | FileCheck -check-prefix=BAREMETAL-AAPCS_VFP %s


// DARWIN-APCS-LABEL: define{{.*}} void @f()
// DARWIN-APCS: call void @g
// DARWIN-AAPCS-LABEL: define{{.*}} arm_aapcscc void @f()
// DARWIN-AAPCS: call arm_aapcscc void @g
// LINUX-APCS-LABEL: define{{.*}} arm_apcscc void @f()
// LINUX-APCS: call arm_apcscc void @g
// LINUX-AAPCS-LABEL: define{{.*}} void @f()
// LINUX-AAPCS: call void @g
// BAREMETAL-AAPCS_VFP-LABEL: define{{.*}} void @f()
// BAREMETAL-AAPCS_VFP: call void @g
// BAREMETAL-AAPCS_VFP: declare void @g()
void g(void);
void f(void) {
  g();
}
