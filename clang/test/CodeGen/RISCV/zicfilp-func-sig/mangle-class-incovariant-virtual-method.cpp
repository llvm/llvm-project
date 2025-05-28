// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-cpu generic-rv64 -target-feature \
// RUN:  +experimental-zicfilp -fcf-protection=branch \
// RUN:  -mcf-branch-label-scheme=func-sig -emit-llvm -o - -x c++ %s \
// RUN:  | FileCheck %s

class Class {
public:
  // test - virtual methods with return type that cannot be covariant mangle
  //        return type as it is declared
  // CHECK-LABEL: define{{.*}} @_ZN5Class34virtualMethodWithIncovariantReturnEv
  // CHECK-SAME: ({{.*}}){{.* !riscv_lpad_func_sig}} [[SIG_MD:![0-9]+]]
  // CHECK-SAME: {{.* !riscv_lpad_label}} [[LB_MD:![0-9]+]] {{.*}}{
  // CHECK-DAG: [[SIG_MD]] = !{!"M1vFivE"}
  // CHECK-DAG: [[LB_MD]] = !{i32 910118}
  //
  virtual int virtualMethodWithIncovariantReturn() { return 0; }
};

void use() { Class C; }
