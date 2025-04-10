// RUN: %clang_cc1 -fopenacc -emit-cir -fclangir %s -o - | FileCheck %s

void acc_shutdown(void) {
  // CHECK: cir.func @acc_shutdown() {
#pragma acc shutdown
// CHECK-NEXT: acc.shutdown loc(#{{[a-zA-Z0-9]+}}){{$}}

#pragma acc shutdown device_type(*)
  // CHECK-NEXT: acc.shutdown attributes {device_types = [#acc.device_type<star>]}
#pragma acc shutdown device_type(nvidia)
  // CHECK-NEXT: acc.shutdown attributes {device_types = [#acc.device_type<nvidia>]}
#pragma acc shutdown device_type(host, multicore)
  // CHECK-NEXT: acc.shutdown attributes {device_types = [#acc.device_type<host>, #acc.device_type<multicore>]}
#pragma acc shutdown device_type(NVIDIA)
  // CHECK-NEXT: acc.shutdown attributes {device_types = [#acc.device_type<nvidia>]}
#pragma acc shutdown device_type(HoSt, MuLtIcORe)
  // CHECK-NEXT: acc.shutdown attributes {device_types = [#acc.device_type<host>, #acc.device_type<multicore>]}
#pragma acc shutdown device_type(HoSt) device_type(MuLtIcORe)
  // CHECK-NEXT: acc.shutdown attributes {device_types = [#acc.device_type<host>, #acc.device_type<multicore>]}
}
