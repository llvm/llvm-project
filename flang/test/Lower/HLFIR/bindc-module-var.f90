! Test BIND(C) module variable lowering
! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

module some_c_module
  integer, bind(c, name="i_var") :: i = 1
  integer, bind(c, name="i_var_no_init") :: i_no_init
  integer, bind(c) :: j_var = 2
  integer, bind(c) :: j_var_no_init
end module

! CHECK-LABEL:   fir.global @i_var : i32 {
! CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i32
! CHECK:           fir.has_value %[[VAL_0]] : i32
! CHECK:         }

! CHECK-LABEL:   fir.global common @i_var_no_init : i32 {
! CHECK:           %[[VAL_0:.*]] = fir.zero_bits i32
! CHECK:           fir.has_value %[[VAL_0]] : i32
! CHECK:         }

! CHECK-LABEL:   fir.global @j_var : i32 {
! CHECK:           %[[VAL_0:.*]] = arith.constant 2 : i32
! CHECK:           fir.has_value %[[VAL_0]] : i32
! CHECK:         }

! CHECK-LABEL:   fir.global common @j_var_no_init : i32 {
! CHECK:           %[[VAL_0:.*]] = fir.zero_bits i32
! CHECK:           fir.has_value %[[VAL_0]] : i32
! CHECK:         }
