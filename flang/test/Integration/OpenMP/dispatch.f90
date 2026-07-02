!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

!RUN: %flang_fc1 -emit-llvm -fopenmp %s -o - | FileCheck %s

!CHECK-LABEL: define void @_QMfuncsPfoo_variant1()
!CHECK: call ptr @_FortranAioBeginExternalListOutput

!CHECK-LABEL: define void @_QMfuncsPfoo_variant2()
!CHECK: call ptr @_FortranAioBeginExternalListOutput

!CHECK-LABEL: define void @_QMfuncsPfoo_dispatch()
!CHECK: %[[COND:.*]] = load i32, ptr @_QMfuncsEfoo_sub
!CHECK: %[[CMP:.*]] = icmp ne i32 %[[COND]], 0
!CHECK: br i1 %[[CMP]], label %[[IF_TRUE:.*]], label %[[IF_FALSE:.*]]
!CHECK: [[IF_TRUE]]:
!CHECK: call void @_QMfuncsPfoo_variant2()
!CHECK: [[IF_FALSE]]:
!CHECK: call void @_QMfuncsPfoo_variant1()

!CHECK-LABEL: define void @_QQmain()
!CHECK: store i32 0, ptr @_QMfuncsEfoo_sub
!CHECK: br label %omp.dispatch.region
!CHECK: omp.dispatch.region:
!CHECK: call void @_QMfuncsPfoo_dispatch()
!CHECK: br label %omp.region.cont
!CHECK: omp.region.cont:
!CHECK: store i32 1, ptr @_QMfuncsEfoo_sub
!CHECK: br label %omp.dispatch.region2
!CHECK: omp.dispatch.region2:
!CHECK: call void @_QMfuncsPfoo_dispatch()
!CHECK: br label %omp.region.cont1
!CHECK: omp.region.cont1:

module funcs
  implicit none
  logical :: foo_sub

contains

  subroutine foo_variant1()
    print *, "in foo_variant1"
  end subroutine

  subroutine foo_variant2()
    print *, "in foo_variant2"
  end subroutine

  subroutine foo_dispatch()
    if (foo_sub) then
      call foo_variant2()
    else
      call foo_variant1()
    end if
  end subroutine

end module funcs

program dispatch_test
  use funcs
  implicit none

  foo_sub = .false.
  !$omp dispatch
  call foo_dispatch()

  foo_sub = .true.
  !$omp dispatch
  call foo_dispatch()

end program
