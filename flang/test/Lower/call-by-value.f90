! Test for PassBy::Value
! RUN: bbc -emit-fir %s -o - | FileCheck %s
program call_by_value
  interface
     subroutine omp_set_nested(enable) bind(c)
       logical, value :: enable
     end subroutine omp_set_nested
  end interface

  logical do_nested
  do_nested = .FALSE.
  call omp_set_nested(do_nested)
end program call_by_value
!CHECK-LABEL: func @_QQmain()
!CHECK: %[[LOGICAL:.*]] = fir.alloca !fir.logical<4>
!CHECK: %false = arith.constant false
!CHECK: %[[VALUE:.*]] = fir.convert %false : (i1) -> !fir.logical<4>
!CHECK: fir.store %[[VALUE]] to %[[LOGICAL]]
!CHECK: %[[LOAD:.*]] = fir.load %[[LOGICAL]]
!CHECK: fir.call @omp_set_nested(%[[LOAD]]) : {{.*}}
