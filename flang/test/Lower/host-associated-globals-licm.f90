! REQUIRES: module-independent
! RUN: split-file %s %t
! RUN: %flang_fc1 -fsave-main-program -emit-hlfir %t/main.f90 -o - | FileCheck %s --check-prefix=MAIN
! RUN: %flang_fc1 -fno-automatic -emit-hlfir %t/main.f90 -o - | FileCheck %s --check-prefix=MAIN
! RUN: %flang_fc1 -emit-hlfir %t/saved.f90 -o - | FileCheck %s --check-prefix=SAVED
! RUN: %flang_fc1 -emit-hlfir -module-dir %t %t/controls.f90 -o - | FileCheck %s --check-prefix=CONTROLS

! Check that host globals shared by sibling internal procedures carry
! internal_assoc on their HLFIR declarations. Cover explicit and
! initialization-implied SAVE, -fsave-main-program and -fno-automatic.
! Controls check unrelated declarations and ordinary host association.

! MAIN-LABEL: func.func private @_QFPaccumulate(
! MAIN: hlfir.declare {{.*}}uniq_name("_QFEx") fortran_attrs<internal_assoc>
! MAIN-LABEL: func.func private @_QFPbump(
!--- main.f90
program p
  integer :: x
  x=0
  call accumulate()
contains
  subroutine accumulate()
    integer :: i,s
    s=0
    do i=1,3
      call bump()
      s=s+x
    end do
    print *,s
  end subroutine
  subroutine bump()
    x=x+1
  end subroutine
end program
! SAVED-LABEL: func.func private @_QFsavedPaccumulate(
! SAVED-DAG: hlfir.declare {{.*}}uniq_name("_QFsavedEx") fortran_attrs<internal_assoc>
! SAVED-DAG: hlfir.declare {{.*}}uniq_name("_QFsavedEy") fortran_attrs<internal_assoc>
! SAVED-LABEL: func.func private @_QFsavedPbump(
!--- saved.f90
subroutine saved()
  integer, save :: x
  integer :: y=0
  x=0
  y=0
  call accumulate()
contains
  subroutine accumulate()
    integer :: i,s
    s=0
    do i=1,3
      call bump()
      s=s+x+y
    end do
    print *,s
  end subroutine
  subroutine bump()
    x=x+1
    y=y+1
  end subroutine
end subroutine
! CONTROLS-LABEL: func.func @_QPcontrols(
! CONTROLS: hlfir.declare {{.*}}uniq_name("_QFcontrolsEuncaptured") :
! CONTROLS-LABEL: func.func private @_QFcontrolsPinner(
! CONTROLS-DAG: hlfir.declare {{.*}}uniq_name("_QMglobalsEx") :
! CONTROLS-DAG: hlfir.declare {{.*}}uniq_name("_QFcontrolsElocal") fortran_attrs<host_assoc>
!--- controls.f90
module globals
  integer :: x
end module
subroutine controls()
  use globals
  integer, save :: uncaptured
  integer :: local
  uncaptured=1
  local=2
  call inner()
  print *,uncaptured
contains
  subroutine inner()
    print *,local,x
  end subroutine
end subroutine
