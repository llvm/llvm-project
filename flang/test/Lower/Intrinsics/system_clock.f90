! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: system_clock_test
subroutine system_clock_test()
  integer(4) :: c
  integer(8) :: m
  real :: r
  ! CHECK-DAG: %[[c:.*]] = fir.alloca i32 {bindc_name = "c"
  ! CHECK-DAG: %[[m:.*]] = fir.alloca i64 {bindc_name = "m"
  ! CHECK-DAG: %[[r:.*]] = fir.alloca f32 {bindc_name = "r"
  ! CHECK: %[[c4:.*]] = arith.constant 4 : i32
  ! CHECK: %[[Count:.*]] = fir.call @_FortranASystemClockCount(%[[c4]]) : (i32) -> i64
  ! CHECK: %[[Count1:.*]] = fir.convert %[[Count]] : (i64) -> i32
  ! CHECK: fir.store %[[Count1]] to %[[c]] : !fir.ref<i32>
  ! CHECK: %[[c8:.*]] = arith.constant 8 : i32
  ! CHECK: %[[Rate:.*]] = fir.call @_FortranASystemClockCountRate(%[[c8]]) : (i32) -> i64
  ! CHECK: %[[Rate1:.*]] = fir.convert %[[Rate]] : (i64) -> f32
  ! CHECK: fir.store %[[Rate1]] to %[[r]] : !fir.ref<f32>
  ! CHECK: %[[c8_2:.*]] = arith.constant 8 : i32
  ! CHECK: %[[Max:.*]] = fir.call @_FortranASystemClockCountMax(%[[c8_2]]) : (i32) -> i64
  ! CHECK: fir.store %[[Max]] to %[[m]] : !fir.ref<i64>
  call system_clock(c, r, m)
! print*, c, r, m
  ! CHECK-NOT: fir.call
  ! CHECK: %[[c8_3:.*]] = arith.constant 8 : i32
  ! CHECK: %[[Rate:.*]] = fir.call @_FortranASystemClockCountRate(%[[c8_3]]) : (i32) -> i64
  ! CHECK: fir.store %[[Rate]] to %[[m]] : !fir.ref<i64>
  call system_clock(count_rate=m)
  ! CHECK-NOT: fir.call
! print*, m
end subroutine
