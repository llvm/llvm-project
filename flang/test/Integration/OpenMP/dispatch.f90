!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

!RUN: %flang_fc1 -emit-llvm -fopenmp %s -o - | FileCheck %s

!CHECK-LABEL: define void @_QMfuncsPfoo_variant()
!CHECK: call ptr @_FortranAioBeginExternalListOutput

!CHECK-LABEL: define void @_QMfuncsPfoo_dispatch()
!CHECK: call ptr @_FortranAioBeginExternalListOutput

!CHECK-LABEL: define void @_QQmain()
!CHECK: call void @_QMfuncsPfoo_dispatch()
!CHECK: br label %omp.dispatch.region
!CHECK: omp.dispatch.region:
!CHECK: call void @_QMfuncsPfoo_variant()
!CHECK: br label %omp.region.cont
!CHECK: omp.region.cont:

module funcs
  implicit none

contains

  subroutine foo_variant()
    print *, "in foo_variant"
  end subroutine

  subroutine foo_dispatch()
    !$omp declare variant(foo_dispatch:foo_variant) match(construct={dispatch})
    print *, "in foo_dispatch"
  end subroutine

end module funcs

program dispatch_test
  use funcs
  implicit none

  call foo_dispatch()

  !$omp dispatch
  call foo_dispatch()

end program
