// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -ftrap-function=trap_func -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -ftrap-function=trap_func -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -ftrap-function=trap_func -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

extern "C" {
  void normal() {}
  // CIR: cir.func{{.*}}@normal()
  // CIR-NOT: trap_func_name
  // LLVM: define{{.*}}@normal() #[[FUNC_ATTR:.*]] {
  void trap_func(){}
  // CIR: cir.func{{.*}}@trap_func()
  // CIR-NOT: trap_func_name
  // LLVM: define{{.*}}@trap_func() #[[FUNC_ATTR]] {

  void caller() {
    normal();
    // CIR: cir.call{{.*}}normal()
    // CIR-SAME: trap_func_name = "trap_func"
    // LLVM: call void{{.*}} @normal() #[[CALL_ATTR:.*]]
    trap_func();
    // CIR: cir.call{{.*}}trap_func()
    // CIR-SAME: trap_func_name = "trap_func"
    // LLVM: call void{{.*}} @trap_func() #[[CALL_ATTR]]
  }
}

// LLVM: attributes #[[FUNC_ATTR]]
// LLVM-NOT: trap-func-name
// LLVM: attributes #[[CALL_ATTR]]
// LLVM-SAME: "trap-func-name"="trap_func"
