// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-cpu generic-rv64 -target-feature \
// RUN:  +experimental-zicfilp -fcf-protection=branch \
// RUN:  -mcf-branch-label-scheme=func-sig -fcxx-exceptions -fexceptions \
// RUN:  -emit-llvm -o - -x c++ %s | FileCheck %s

// test - `<exception-spec>` should be ignored
// CHECK-LABEL: define{{.*}} @_Z9funcThrowv()
// CHECK-SAME: {{.* !riscv_lpad_func_sig}} [[SIG_MD:![0-9]+]]
// CHECK-SAME: {{.* !riscv_lpad_label}} [[LB_MD:![0-9]+]] {{.*}}{
///
void funcThrow() { throw 0; }
// CHECK-LABEL: define{{.*}} @_Z12funcNoExceptv()
// CHECK-SAME: {{.* !riscv_lpad_func_sig}} [[SIG_MD]]
// CHECK-SAME: {{.* !riscv_lpad_label}} [[LB_MD]] {{.*}}{
//
void funcNoExcept() noexcept {}
