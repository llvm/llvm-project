! Test that logicals are lowered to Fortran logical types where it matters
! RUN: bbc %s -emit-fir -o - | FileCheck %s

! Logicals should be lowered to Fortran logical types in memory/function
! interfaces.


! CHECK-LABEL: _QPtest_value_arguments
subroutine test_value_arguments()
interface
subroutine foo2(l)
  logical(2) :: l
end subroutine
subroutine foo4(l)
  logical(4) :: l
end subroutine
end interface

  ! CHECK: %[[true2:.*]] = fir.convert %true{{.*}} : (i1) -> !fir.logical<2>
  ! CHECK: fir.store %[[true2]] to %[[mem2:.*]] : !fir.ref<!fir.logical<2>>
  ! CHECK: fir.call @_QPfoo2(%[[mem2]]) : (!fir.ref<!fir.logical<2>>) -> ()
call foo2(.true._2)

  ! CHECK: %[[true4:.*]] = fir.convert %true{{.*}} : (i1) -> !fir.logical<4>
  ! CHECK: fir.store %[[true4]] to %[[mem4:.*]] : !fir.ref<!fir.logical<4>>
  ! CHECK: fir.call @_QPfoo4(%[[mem4]]) : (!fir.ref<!fir.logical<4>>) -> ()
call foo4(.true.)

end subroutine
