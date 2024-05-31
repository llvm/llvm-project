// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-cpu generic-rv64 -target-feature \
// RUN:  +experimental-zicfilp -fcf-protection=branch \
// RUN:  -mcf-branch-label-scheme=func-sig -emit-llvm -o - -x c++ %s \
// RUN:  | FileCheck %s

// test - `wchar_t` in C++ should be mangled to `wchar_t` in C
// CHECK-LABEL: define{{.*}} @_Z14funcWithWCharTw({{.*}})
// CHECK-SAME: {{.* !riscv_lpad_func_sig}} [[SIG_MD:![0-9]+]]
// CHECK-SAME: {{.* !riscv_lpad_label}} [[LB_MD:![0-9]+]] {{.*}}{
// CHECK-DAG: [[SIG_MD]] = !{!"FviE"}
// CHECK-DAG: [[LB_MD]] = !{i32 374765}
//
void funcWithWCharT(wchar_t) {}
