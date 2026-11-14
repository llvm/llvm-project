! RUN: split-file %s %t
! RUN: %flang_fc1 -fsyntax-only -module-dir %t %t/m.f90
! RUN: %flang_fc1 -emit-hlfir -I %t -mmlir --mlir-print-debuginfo -mmlir --mlir-print-local-scope %t/use.f90 -o - | FileCheck %s

! Check that the declaration of a module procedure that is only accessible
! through a use-associated generic gets the location of the specific procedure
! rather than an unknown location. This covers a USE with an ONLY list that
! names the generic but not the specific, and a PRIVATE specific behind a
! PUBLIC generic.

!--- m.f90
module callee_loc_m
  interface get
    module procedure nxt
  end interface
contains
  function nxt(i)
    integer, intent(in) :: i
    integer :: nxt
    nxt = i + 1
  end function
end module

module callee_loc_p
  private
  public :: getp
  interface getp
    module procedure nxtp
  end interface
contains
  function nxtp(i)
    integer, intent(in) :: i
    integer :: nxtp
    nxtp = i + 2
  end function
end module

!--- use.f90
subroutine s(g, v)
  use callee_loc_m, only: get
  integer :: g, v
  v = get(g)
end subroutine

subroutine sp(g, v)
  use callee_loc_p
  integer :: g, v
  v = getp(g)
end subroutine

! CHECK: func.func private @_QMcallee_loc_mPnxt({{.*}}) -> i32 loc("{{.*}}callee_loc_m.mod":7:10)
! CHECK: func.func private @_QMcallee_loc_pPnxtp({{.*}}) -> i32 loc("{{.*}}callee_loc_p.mod":8:10)
