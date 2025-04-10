// RUN: %clang_cc1 -fopenacc -emit-cir -fclangir %s -o - | FileCheck %s

void acc_set(void) {
  // CHECK: cir.func @acc_set() {

#pragma acc set device_type(*)
  // CHECK-NEXT: acc.set attributes {device_type = #acc.device_type<star>}

  // Set doesn't allow multiple device_type clauses, so no need to test them.
#pragma acc set device_type(radeon)
  // CHECK-NEXT: acc.set attributes {device_type = #acc.device_type<radeon>}

  // CHECK-NEXT: cir.return
}
