// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-cpu generic-rv64 -target-feature \
// RUN:  +experimental-zicfilp -fcf-protection=branch \
// RUN:  -mcf-branch-label-scheme=func-sig -emit-llvm -o - -x c++ %s \
// RUN:  | FileCheck %s --check-prefixes=PTR,LREF,RREF

class Class {
public:
  // test - virtual methods with return type that can possibly be covariant
  //        mangle return class as `class v`
  // PTR-LABEL: define{{.*}} @_ZN5Class35virtualMethodWithCovariantPtrReturnEv
  // PTR-SAME: ({{.*}}){{.* !riscv_lpad_func_sig}} [[SIG_MD_PTR:![0-9]+]]
  // PTR-SAME: {{.* !riscv_lpad_label}} [[LB_MD_PTR:![0-9]+]] {{.*}}{
  //
  virtual Class *virtualMethodWithCovariantPtrReturn() { return this; }

  // LREF-LABEL: define{{.*}} @_ZN5Class36virtualMethodWithCovariantLRefReturnEv
  // LREF-SAME: ({{.*}}){{.* !riscv_lpad_func_sig}} [[SIG_MD_LREF:![0-9]+]]
  // LREF-SAME: {{.* !riscv_lpad_label}} [[LB_MD_LREF:![0-9]+]] {{.*}}{
  //
  virtual Class &virtualMethodWithCovariantLRefReturn() { return *this; }

  // RREF-LABEL: define{{.*}} @_ZN5Class36virtualMethodWithCovariantRRefReturnEv
  // RREF-SAME: ({{.*}}){{.* !riscv_lpad_func_sig}} [[SIG_MD_RREF:![0-9]+]]
  // RREF-SAME: {{.* !riscv_lpad_label}} [[LB_MD_RREF:![0-9]+]] {{.*}}{
  //
  virtual Class &&virtualMethodWithCovariantRRefReturn() {
    return static_cast<Class&&>(*this);
  }
};

// PTR-DAG: [[SIG_MD_PTR]] = !{!"M1vFP1vvE"}
// PTR-DAG: [[LB_MD_PTR]] = !{i32 996333}
// LREF-DAG: [[SIG_MD_LREF]] = !{!"M1vFR1vvE"}
// LREF-DAG: [[LB_MD_LREF]] = !{i32 918198}
// RREF-DAG: [[SIG_MD_RREF]] = !{!"M1vFO1vvE"}
// RREF-DAG: [[LB_MD_RREF]] = !{i32 86168}

void use() { Class C; }
