// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -disable-llvm-passes -Os -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -disable-llvm-passes -Os -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=BOTH
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -disable-llvm-passes -Os -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=BOTH
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -disable-llvm-passes -Oz -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR,CIROZ
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -disable-llvm-passes -Oz -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=BOTH,BOTHOZ
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -disable-llvm-passes -Oz -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=BOTH,BOTHOZ

extern "C" {
  __attribute__((hot))
  void normal(){}
  // CIR: cir.func{{.*}}@normal() 
  // CIROZ-SAME: minsize
  // CIR-SAME: optsize
  // BOTH: define{{.*}}@normal(){{.*}} #[[NORMAL_ATTR:.*]] {

  __attribute__((cold))
  __attribute__((optnone))
  void optnone(){}
  // CIR: cir.func{{.*}}@optnone() 
  // CIR-NOT: optsize
  // CIR-NOT: minsize
  // BOTH: define{{.*}}@optnone(){{.*}} #[[OPTNONE_ATTR:.*]] {

  // CIR: cir.func{{.*}}@caller()
  void caller() {
    normal();
    // CIR: cir.call{{.*}}@normal()
    // CIROZ-SAME: minsize
    // CIR-SAME: optsize
    // BOTH: call void @normal() #[[NORMAL_CALL_ATTR:.*]]
    optnone();
    // CIR: cir.call{{.*}}@optnone()
    // CIR-NOT: optsize
    // CIR-NOT: minsize
    // BOTH: call void @optnone() #[[OPTNONE_CALL_ATTR:.*]]

    // CIR: cir.return
  }
}

// BOTH: attributes #[[NORMAL_ATTR]]
// BOTHOZ-SAME: minsize
// BOTH-SAME: optsize
//
// BOTH: attributes #[[OPTNONE_ATTR]]
// BOTH-NOT: optsize
// BOTH-NOT: minsize
//
// attributes for caller, to block the 'NOT'.
// BOTH: attributes
//
// BOTH: attributes #[[NORMAL_CALL_ATTR]]
// BOTHOZ-SAME: minsize
// BOTH-SAME: optsize
//
// BOTH: attributes #[[OPTNONE_CALL_ATTR]]
// BOTH-NOT: optsize
// BOTH-NOT: minsize
//
// to block the 'NOT'.
// BOTH: !llvm.
