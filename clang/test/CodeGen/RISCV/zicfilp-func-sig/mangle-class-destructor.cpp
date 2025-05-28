// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-cpu generic-rv64 -target-feature \
// RUN:  +experimental-zicfilp -fcf-protection=branch \
// RUN:  -mcf-branch-label-scheme=func-sig -emit-llvm -o - -x c++ %s \
// RUN:  | FileCheck %s

class Class {
public:
  // test - destructors should use `void (*)(void*)`
  // CHECK-LABEL: define{{.*}} @_ZN5ClassD1Ev({{.*}})
  // CHECK-SAME: {{.* !riscv_lpad_func_sig}} [[SIG_MD:![0-9]+]]
  // CHECK-SAME: {{.* !riscv_lpad_label}} [[LB_MD:![0-9]+]] {{.*}}{
  // CHECK-DAG: [[SIG_MD]] = !{!"FvPvE"}
  // CHECK-DAG: [[LB_MD]] = !{i32 408002}
  //
  ~Class() {}
};

void use() { Class C; }
