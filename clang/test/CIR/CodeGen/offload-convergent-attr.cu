// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fcuda-is-device -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fcuda-is-device -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fcuda-is-device -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

extern "C" {
__attribute__((device))
void normal() {}
// CIR: cir.func{{.*}}@normal()
// CIR-SAME: convergent
// LLVM: define {{.*}}@normal(){{.*}} #[[NORMAL_ATTR:.*]] {

__attribute__((hot))
__attribute__((device))
__attribute__((noconvergent))
void no_conv() {}
// CIR: cir.func{{.*}}@no_conv()
// CIR-NOT: convergent
// LLVM: define {{.*}}@no_conv(){{.*}} #[[NO_CONV_ATTR:.*]] {

// CIR: cir.func{{.*}}@caller
__attribute__((device))
void caller() {
  normal();
  // CIR: cir.call{{.*}}@normal()
  // CIR-SAME: convergent
  // LLVM: call void{{.*}}@normal() #[[NORMAL_CALL_ATTR:.*]]
  no_conv();
  // CIR: cir.call{{.*}}@no_conv()
  // CIR-NOT: convergent
  // CIR: cir.return
  // LLVM: call void{{.*}}@no_conv() #[[NO_CONV_CALL_ATTR:.*]]
}
}

// LLVM: attributes #[[NORMAL_ATTR]] 
// LLVM-SAME: convergent
// LLVM: attributes #[[NO_CONV_ATTR]] 
// LLVM-NOT: convergent
// LLVM: attributes #[[NORMAL_CALL_ATTR]] 
// LLVM-SAME: convergent
// LLVM: attributes #[[NO_CONV_CALL_ATTR]] 
// LLVM-NOT: convergent
