// RUN: %clang_cc1 -fopenacc -emit-cir -fclangir %s -o - | FileCheck %s

void acc_init(void) {
  // CHECK: cir.func @acc_init() {
#pragma acc init
// CHECK-NEXT: acc.init loc(#{{[a-zA-Z0-9]+}}){{$}}

#pragma acc init device_type(*)
  // CHECK-NEXT: acc.init attributes {device_types = [#acc.device_type<star>]}
#pragma acc init device_type(nvidia)
  // CHECK-NEXT: acc.init attributes {device_types = [#acc.device_type<nvidia>]}
#pragma acc init device_type(host, multicore)
  // CHECK-NEXT: acc.init attributes {device_types = [#acc.device_type<host>, #acc.device_type<multicore>]}
#pragma acc init device_type(NVIDIA)
  // CHECK-NEXT: acc.init attributes {device_types = [#acc.device_type<nvidia>]}
#pragma acc init device_type(HoSt, MuLtIcORe)
  // CHECK-NEXT: acc.init attributes {device_types = [#acc.device_type<host>, #acc.device_type<multicore>]}
#pragma acc init device_type(HoSt) device_type(MuLtIcORe)
  // CHECK-NEXT: acc.init attributes {device_types = [#acc.device_type<host>, #acc.device_type<multicore>]}
}
