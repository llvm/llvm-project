// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -msave-reg-params -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -msave-reg-params -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -msave-reg-params -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

extern "C" {

  __attribute__((hot))
  void func(){}
  // CIR: cir.func{{.*}}@func()
  // CIR-SAME: save_reg_params
  // LLVM: define{{.*}}@func() #[[FUNC_ATTRS:.*]] {

  void caller() {
    func();
    // CIR: cir.call{{.*}}@func()
    // CIR-NOT: save_reg_params
    // CIR: cir.return
    // LLVM: call void{{.*}}@func() #[[CALL_ATTRS:.*]] 

  }
}

// LLVM: attributes #[[FUNC_ATTRS]]
// LLVM-SAME: "save-reg-params"
// LLVM: attributes #[[CALL_ATTRS]]
// LLVM-NOT: "save-reg-params"
