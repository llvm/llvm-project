// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-cpu generic-rv64 -target-feature \
// RUN:  +experimental-zicfilp -fcf-protection=branch \
// RUN:  -mcf-branch-label-scheme=func-sig -emit-llvm -o - -x c++ %s \
// RUN:  | FileCheck %s

// CHECK-LABEL: define{{.*}} @_Z13nonMemberFuncv()
// CHECK-SAME: {{.* !riscv_lpad_func_sig}} [[SIG_MD:![0-9]+]]
// CHECK-SAME: {{.* !riscv_lpad_label}} [[LB_MD:![0-9]+]] {{.*}}{
//
void nonMemberFunc() {}

class Class {
public:
  // test - static methods are mangled as non-member function
  // CHECK-LABEL: define{{.*}} @_ZN5Class12staticMethodEv()
  // CHECK-SAME: {{.* !riscv_lpad_func_sig}} [[SIG_MD]]
  // CHECK-SAME: {{.* !riscv_lpad_label}} [[LB_MD]] {{.*}}{
  //
  static void staticMethod() {}
};

void use() { Class::staticMethod(); }
