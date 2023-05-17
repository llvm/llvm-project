! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s
! RUN: bbc -emit-fir -o - %s | FileCheck %s

program test
  continue
end

! Test that a null pointer is generated for environment defaults if nothing is specified

! CHECK: fir.global @_QQEnvironmentDefaults constant : !fir.ref<tuple<i[[int_size:.*]], !fir.ref<!fir.array<0xtuple<!fir.ref<i8>, !fir.ref<i8>>>>>> {
! CHECK:  %[[VAL_0:.*]] = fir.zero_bits !fir.ref<tuple<i[[int_size]], !fir.ref<!fir.array<0xtuple<!fir.ref<i8>, !fir.ref<i8>>>>>>
! CHECK: fir.has_value  %[[VAL_0]] : !fir.ref<tuple<i[[int_size]], !fir.ref<!fir.array<0xtuple<!fir.ref<i8>, !fir.ref<i8>>>>>>
