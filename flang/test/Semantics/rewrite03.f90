!RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
!Test rewriting of misparsed statement function definitions
!into array element assignment statements.

program main
  real sf(1)
  integer :: j = 1
!CHECK: sf(int(j,kind=8))=1._4
  sf(j) = 1.
end

function func
  real sf(1)
  integer :: j = 1
!CHECK: sf(int(j,kind=8))=2._4
  sf(j) = 2.
  func = 0.
end

subroutine subr
  real sf(1)
  integer :: j = 1
!CHECK: sf(int(j,kind=8))=3._4
  sf(j) = 3.
end

module m
  interface
    module subroutine smp
    end
  end interface
end
submodule(m) sm
 contains
  module procedure smp
    real sf(1)
    integer :: j = 1
!CHECK: sf(int(j,kind=8))=4._4
    sf(j) = 4.
  end
end

subroutine block
  block
    real sf(1)
    integer :: j = 1
!CHECK: sf(int(j,kind=8))=5._4
    sf(j) = 5.
  end block
end
