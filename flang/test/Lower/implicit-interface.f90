! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPchar_return_callee(%arg0: !fir.ref<!fir.char<1>>, %arg1: index, %arg2: !fir.ref<i32>) -> !fir.boxchar<1>
function char_return_callee(i)
  character(10) :: char_return_callee
  integer :: i
end function

! CHECK-LABEL: @_QPtest_char_return_caller()
subroutine test_char_return_caller
  character(10) :: char_return_caller
  ! CHECK: fir.call @_QPchar_return_caller({{.*}}) : (!fir.ref<!fir.char<1>>, index, !fir.ref<i32>) -> !fir.boxchar<1>
  print *, char_return_caller(5)
end subroutine

! CHECK-LABEL: func @_QPtest_passing_char_array()
subroutine test_passing_char_array
  character(len=3) :: x(4)
  call sub_taking_a_char_array(x)
  ! CHECK-DAG: %[[xarray:.*]] = fir.alloca !fir.array<3x4x!fir.char<1>>
  ! CHECK-DAG: %[[c3:.*]] = constant 3 : index
  ! CHECK-DAG: %[[xbuff:.*]] = fir.convert %[[xarray]] : (!fir.ref<!fir.array<3x4x!fir.char<1>>>) -> !fir.ref<!fir.char<1>>
  ! CHECK: %[[boxchar:.*]] = fir.emboxchar %[[xbuff]], %[[c3]] : (!fir.ref<!fir.char<1>>, index) -> !fir.boxchar<1>
  ! CHECK: fir.call @_QPsub_taking_a_char_array(%[[boxchar]]) : (!fir.boxchar<1>) -> () 
end subroutine

! TODO more implicit interface cases with/without explicit interface

