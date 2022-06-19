! Test various aspects around call lowering. More detailed tests around core
! requirements are done in call-xxx.f90 and dummy-argument-xxx.f90 files.

! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPtest_nested_calls
subroutine test_nested_calls()
  interface
    subroutine foo(i)
      integer :: i
    end subroutine
    integer function bar()
    end function
  end interface
  ! CHECK: %[[result_storage:.*]] = fir.alloca i32 {adapt.valuebyref}
  ! CHECK: %[[result:.*]] = fir.call @_QPbar() : () -> i32
  ! CHECK: fir.store %[[result]] to %[[result_storage]] : !fir.ref<i32>
  ! CHECK: fir.call @_QPfoo(%[[result_storage]]) : (!fir.ref<i32>) -> ()
  call foo(bar())
end subroutine
