! Test the `-emit-mlir` option

! RUN: %flang_fc1 -emit-mlir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

! Verify that an `.mlir` file is created when `-emit-mlir` is used. Do it in a temporary directory, which will be cleaned up by the
! LIT runner.
! RUN: rm -rf %t-dir && mkdir -p %t-dir && cd %t-dir
! RUN: cp %s .
! RUN: %flang_fc1 -emit-mlir emit-mlir.f90 && ls emit-mlir.mlir

! CHECK: module attributes {
! CHECK-LABEL: func @_QQmain() {
! CHECK-NEXT:  return
! CHECK-NEXT: }
! CHECK-NEXT: fir.global @_QQEnvironmentDefaults constant : !fir.ref<tuple<i[[int_size:.*]], !fir.ref<!fir.array<0xtuple<!fir.ref<i8>, !fir.ref<i8>>>>>> {
! CHECK-NEXT:  %[[VAL_0:.*]] = fir.zero_bits !fir.ref<tuple<i[[int_size]], !fir.ref<!fir.array<0xtuple<!fir.ref<i8>, !fir.ref<i8>>>>>>
! CHECK-NEXT: fir.has_value  %[[VAL_0]] : !fir.ref<tuple<i[[int_size]], !fir.ref<!fir.array<0xtuple<!fir.ref<i8>, !fir.ref<i8>>>>>>
! CHECK-NEXT: }
! CHECK-NEXT: }

end program
