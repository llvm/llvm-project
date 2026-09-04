// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR,CIR_NONE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM,LLVM_NONE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM,LLVM_NONE

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fzero-call-used-regs=skip -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR,CIR_SKIP
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fzero-call-used-regs=skip -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM,LLVM_SKIP
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fzero-call-used-regs=skip -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM,LLVM_SKIP

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fzero-call-used-regs=all-gpr -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR,CIR_ALLGPR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fzero-call-used-regs=all-gpr -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM,LLVM_ALLGPR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fzero-call-used-regs=all-gpr -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM,LLVM_ALLGPR

extern "C" {
  __attribute__((hot))
  void normal(){}
  // CIR: cir.func{{.*}}@normal()
  // CIR_NONE-NOT: zero-call-used-regs
  // CIR_SKIP-NOT: zero-call-used-regs
  // CIR_ALLGPR-SAME: zero_call_used_regs = "all-gpr"
  // LLVM: define{{.*}}@normal() #[[NORM_ATTR:.*]] {

  __attribute__((cold))
  __attribute__((zero_call_used_regs("skip")))
  void skip() { }
  // CIR: cir.func{{.*}}@skip()
  // CIR-SAME: zero_call_used_regs = "skip"
  // LLVM: define{{.*}}@skip() #[[SKIP_ATTR:.*]] {

  __attribute__((zero_call_used_regs("all")))
  void all() { }
  // CIR: cir.func{{.*}}@all()
  // CIR-SAME: zero_call_used_regs = "all"
  // LLVM: define{{.*}}@all() #[[ALL_ATTR:.*]] {

  __attribute__((zero_call_used_regs("used")))
  void used() { }
  // CIR: cir.func{{.*}}@used()
  // CIR-SAME: zero_call_used_regs = "used"
  // LLVM: define{{.*}}@used() #[[USED_ATTR:.*]] {

  __attribute__((zero_call_used_regs("used-gpr-arg")))
  void used_gpr_arg() { }
  // CIR: cir.func{{.*}}@used_gpr_arg()
  // CIR-SAME: zero_call_used_regs = "used-gpr-arg"
  // LLVM: define{{.*}}@used_gpr_arg() #[[USED_GPR_ATTR:.*]] {

  void caller() {
    normal();
    // CIR: cir.call{{.*}}@normal()
    // CIR-NOT: zero-call-used-regs
    // LLVM: call void{{.*}}@normal() #[[NORM_CALL_ATTR:.*]]
    skip();
    // CIR: cir.call{{.*}}@skip()
    // CIR-SAME: zero_call_used_regs = "skip"
    // LLVM: call void{{.*}}@skip() #[[SKIP_CALL_ATTR:.*]]
    all();
    // CIR: cir.call{{.*}}@all()
    // CIR-SAME: zero_call_used_regs = "all"
    // LLVM: call void{{.*}}@all() #[[ALL_CALL_ATTR:.*]]
    used();
    // CIR: cir.call{{.*}}@used()
    // CIR-SAME: zero_call_used_regs = "used"
    // LLVM: call void{{.*}}@used() #[[USED_CALL_ATTR:.*]]
    used_gpr_arg();
    // CIR: cir.call{{.*}}@used_gpr_arg()
    // CIR-SAME: zero_call_used_regs = "used-gpr-arg"
    // LLVM: call void{{.*}}@used_gpr_arg() #[[USED_GPR_CALL_ATTR:.*]]
  }
}

// LLVM: attributes #[[NORM_ATTR]]
// LLVM_NONE-NOT: zero-call-used-regs
// LLVM_SKIP-NOT: zero-call-used-regs
// LLVM_ALLGPR-SAME: "zero-call-used-regs"="all-gpr"
// LLVM: attributes #[[SKIP_ATTR]]
// LLVM-SAME: "zero-call-used-regs"="skip"
// LLVM: attributes #[[ALL_ATTR]]
// LLVM-SAME: "zero-call-used-regs"="all"
// LLVM: attributes #[[USED_ATTR]]
// LLVM-SAME: "zero-call-used-regs"="used"
// LLVM: attributes #[[USED_GPR_ATTR]]
// LLVM-SAME: "zero-call-used-regs"="used-gpr-arg"
//
// LLVM: attributes #[[NORM_CALL_ATTR]]
// LLVM-NOT: zero-call-used-regs
// LLVM: attributes #[[SKIP_CALL_ATTR]]
// LLVM-SAME: "zero-call-used-regs"="skip"
// LLVM: attributes #[[ALL_CALL_ATTR]]
// LLVM-SAME: "zero-call-used-regs"="all"
// LLVM: attributes #[[USED_CALL_ATTR]]
// LLVM-SAME: "zero-call-used-regs"="used"
// LLVM: attributes #[[USED_GPR_CALL_ATTR]]
// LLVM-SAME: "zero-call-used-regs"="used-gpr-arg"
