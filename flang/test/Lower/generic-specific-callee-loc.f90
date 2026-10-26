! RUN: split-file %s %t
! RUN: bbc -emit-hlfir %t/m.f90 --module=%t -o /dev/null && bbc -emit-hlfir %t/use.f90 -I %t -o - --mlir-print-debuginfo | FileCheck %s

! Check that the declaration of a module procedure that is only accessible
! through a use-associated generic (USE with an ONLY list that names the
! generic but not the specific) gets the location of the specific procedure
! rather than an unknown location.

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

!--- use.f90
subroutine s(g, v)
  use callee_loc_m, only: get
  integer :: g, v
  v = get(g)
end subroutine

! CHECK: func.func private @_QMcallee_loc_mPnxt({{.*}}) -> i32 loc("{{.*}}callee_loc_m.mod":7:10)
