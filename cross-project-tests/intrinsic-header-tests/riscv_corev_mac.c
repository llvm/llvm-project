// REQUIRES: riscv-registered-target
// expected-no-diagnostics

// RUN: %clang %s -O2 -S -o - --target=riscv32 -march=rv32i_xcvmac \
// RUN:   -Werror -Wextra -Xclang -verify \
// RUN:   | FileCheck %s

#include <riscv_corev_mac.h>

// CHECK-LABEL: test_mac_mac:
// CHECK: cv.mac
int32_t test_mac_mac(int32_t a0, int32_t a1, int32_t a2) {
  return __riscv_cv_mac_mac(a0, a1, a2);
}

// CHECK-LABEL: test_mac_msu:
// CHECK: cv.msu
int32_t test_mac_msu(int32_t a0, int32_t a1, int32_t a2) {
  return __riscv_cv_mac_msu(a0, a1, a2);
}

// CHECK-LABEL: test_mac_muluN:
// CHECK: cv.mulun {{.*}}, 1
uint32_t test_mac_muluN(uint32_t a0, uint32_t a1) {
  return __riscv_cv_mac_muluN(a0, a1, 1);
}

// CHECK-LABEL: test_mac_mulhhuN:
// CHECK: cv.mulhhun {{.*}}, 1
uint32_t test_mac_mulhhuN(uint32_t a0, uint32_t a1) {
  return __riscv_cv_mac_mulhhuN(a0, a1, 1);
}

// CHECK-LABEL: test_mac_mulsN:
// CHECK: cv.mulsn {{.*}}, 1
int32_t test_mac_mulsN(uint32_t a0, uint32_t a1) {
  return __riscv_cv_mac_mulsN(a0, a1, 1);
}

// CHECK-LABEL: test_mac_mulhhsN:
// CHECK: cv.mulhhsn {{.*}}, 1
int32_t test_mac_mulhhsN(uint32_t a0, uint32_t a1) {
  return __riscv_cv_mac_mulhhsN(a0, a1, 1);
}

// CHECK-LABEL: test_mac_muluRN:
// CHECK: cv.mulurn {{.*}}, 1
uint32_t test_mac_muluRN(uint32_t a0, uint32_t a1) {
  return __riscv_cv_mac_muluRN(a0, a1, 1);
}

// CHECK-LABEL: test_mac_mulhhuRN:
// CHECK: cv.mulhhurn {{.*}}, 1
uint32_t test_mac_mulhhuRN(uint32_t a0, uint32_t a1) {
  return __riscv_cv_mac_mulhhuRN(a0, a1, 1);
}

// CHECK-LABEL: test_mac_mulsRN:
// CHECK: cv.mulsrn {{.*}}, 1
int32_t test_mac_mulsRN(uint32_t a0, uint32_t a1) {
  return __riscv_cv_mac_mulsRN(a0, a1, 1);
}

// CHECK-LABEL: test_mac_mulhhsRN:
// CHECK: cv.mulhhsrn {{.*}}, 1
int32_t test_mac_mulhhsRN(uint32_t a0, uint32_t a1) {
  return __riscv_cv_mac_mulhhsRN(a0, a1, 1);
}

// CHECK-LABEL: test_mac_macuN:
// CHECK: cv.macun {{.*}}, 1
uint32_t test_mac_macuN(uint32_t a0, uint32_t a1, uint32_t a2) {
  return __riscv_cv_mac_macuN(a0, a1, a2, 1);
}

// CHECK-LABEL: test_mac_machhuN:
// CHECK: cv.machhun {{.*}}, 1
uint32_t test_mac_machhuN(uint32_t a0, uint32_t a1, uint32_t a2) {
  return __riscv_cv_mac_machhuN(a0, a1, a2, 1);
}

// CHECK-LABEL: test_mac_macsN:
// CHECK: cv.macsn {{.*}}, 1
int32_t test_mac_macsN(uint32_t a0, uint32_t a1, uint32_t a2) {
  return __riscv_cv_mac_macsN(a0, a1, a2, 1);
}

// CHECK-LABEL: test_mac_machhsN:
// CHECK: cv.machhsn {{.*}}, 1
int32_t test_mac_machhsN(uint32_t a0, uint32_t a1, uint32_t a2) {
  return __riscv_cv_mac_machhsN(a0, a1, a2, 1);
}

// CHECK-LABEL: test_mac_macuRN:
// CHECK: cv.macurn {{.*}}, 1
uint32_t test_mac_macuRN(uint32_t a0, uint32_t a1, uint32_t a2) {
  return __riscv_cv_mac_macuRN(a0, a1, a2, 1);
}

// CHECK-LABEL: test_mac_machhuRN:
// CHECK: cv.machhurn {{.*}}, 1
uint32_t test_mac_machhuRN(uint32_t a0, uint32_t a1, uint32_t a2) {
  return __riscv_cv_mac_machhuRN(a0, a1, a2, 1);
}

// CHECK-LABEL: test_mac_macsRN:
// CHECK: cv.macsrn {{.*}}, 1
int32_t test_mac_macsRN(uint32_t a0, uint32_t a1, uint32_t a2) {
  return __riscv_cv_mac_macsRN(a0, a1, a2, 1);
}

// CHECK-LABEL: test_mac_machhsRN:
// CHECK: cv.machhsrn {{.*}}, 1
int32_t test_mac_machhsRN(uint32_t a0, uint32_t a1, uint32_t a2) {
  return __riscv_cv_mac_machhsRN(a0, a1, a2, 1);
}
