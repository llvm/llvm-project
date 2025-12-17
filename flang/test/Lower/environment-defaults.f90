! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s
! RUN: bbc -emit-fir -o - %s | FileCheck %s

program test
  continue
end

! Test that a null pointer is passed for environment defaults if nothing is specified

! CHECK-NOT: @_QQEnvironmentDefaults

! CHECK: %0 = fir.zero_bits !fir.ref<tuple<i32, !fir.ref<!fir.array<0xtuple<!fir.ref<i8>, !fir.ref<i8>>>>>>
! CHECK-NEXT: @_FortranAProgramStart(%arg0, %arg1, %arg2, %0)
