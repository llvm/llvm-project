!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

! RUN: %flang_fc1 -fsafe-trampoline -emit-llvm -o - %s | FileCheck %s
!
! A module procedure pointer with an initializer is a global whose initializer
! emboxes a procedure. With -fsafe-trampoline the boxed-procedure pass used to
! leave that fir.emboxproc in place, and codegen aborted with
! "not yet implemented: fir.emboxproc codegen". No internal procedure or
! trampoline is involved: the globals must simply lower to plain pointers.

! CHECK: @_QMmEnull_ptr = {{.*}}global ptr null
! CHECK: @_QMmEtarget_ptr = {{.*}}global ptr @_QMmPcallee
! CHECK-NOT: TrampolineInit

module m
  implicit none
  abstract interface
    subroutine callback()
    end subroutine
  end interface
  procedure(callback), pointer :: null_ptr => null()
  procedure(callback), pointer :: target_ptr => callee
contains
  subroutine callee()
  end subroutine
end module
