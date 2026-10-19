! REQUIRES: module-independent
! RUN: split-file %s %t
! RUN: %flang_fc1 -fsave-main-program -emit-fir %t/main.f90 -o - | fir-opt --flang-licm | FileCheck %s --check-prefix=MAIN
! RUN: %flang_fc1 -fno-automatic -emit-fir %t/main.f90 -o - | fir-opt --flang-licm | FileCheck %s --check-prefix=MAIN
! RUN: %flang_fc1 -emit-fir %t/saved.f90 -o - | fir-opt --flang-licm | FileCheck %s --check-prefix=SAVED
! RUN: %flang_fc1 -emit-hlfir -module-dir %t %t/controls.f90 -o - | FileCheck %s --check-prefix=CONTROLS

! Check that host globals shared by sibling internal procedures carry
! internal_assoc. Without it, FIR alias analysis may miss writes by bump(),
! allowing FIR LICM to hoist loads out of accumulate's loop. Loads must
! remain after the call for explicit/initialization-implied SAVE and for
! -fsave-main-program/-fno-automatic. Controls check that unrelated declarations
! and ordinary host association keep their existing attributes.

! MAIN-LABEL: func.func private @_QFPaccumulate(
! MAIN: %[[X:[^ ]+]] = fir.declare {{.*}}uniq_name("_QFEx") fortran_attrs<internal_assoc>
! MAIN: fir.do_loop
! MAIN-NOT: {{^ *[}]}}
! MAIN: fir.call @_QFPbump(
! MAIN-NOT: {{^ *[}]}}
! MAIN: fir.load %[[X]] :
! MAIN: {{^ *[}]}}
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
! SAVED-DAG: %[[X:[^ ]+]] = fir.declare {{.*}}uniq_name("_QFsavedEx") fortran_attrs<internal_assoc>
! SAVED-DAG: %[[Y:[^ ]+]] = fir.declare {{.*}}uniq_name("_QFsavedEy") fortran_attrs<internal_assoc>
! SAVED: fir.do_loop
! SAVED-NOT: {{^ *[}]}}
! SAVED: fir.call @_QFsavedPbump(
! SAVED-NOT: {{^ *[}]}}
! SAVED: fir.load %[[X]] :
! SAVED-NOT: {{^ *[}]}}
! SAVED: fir.load %[[Y]] :
! SAVED: {{^ *[}]}}
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
