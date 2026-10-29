! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp

! OpenMP 5.2 §7.8.2: The bare form of DECLARE TARGET (without arguments or
! clauses) is only permitted in the specification part of a subroutine,
! function, or interface body.

block data foo
  implicit none
  integer :: x
  common /blk/ x
  !ERROR: DECLARE TARGET directive without arguments or clauses must appear in a subroutine or function
  !$omp declare target
  data x /42/
end block data

program main
  implicit none
  integer :: a
  !ERROR: DECLARE TARGET directive without arguments or clauses must appear in a subroutine or function
  !$omp declare target
  a = 1
end program

! Valid: bare form in a subroutine
subroutine sub1()
  implicit none
  integer :: b
  !$omp declare target
  b = 2
end subroutine

! Valid: bare form in a function
integer function func1()
  !$omp declare target
  func1 = 3
end function

module mod1
  implicit none
  integer :: c
  !ERROR: DECLARE TARGET directive without arguments or clauses must appear in a subroutine or function
  !$omp declare target
end module

module parent_mod
  implicit none
  integer :: d
end module

submodule (parent_mod) child_sub
  integer :: e
  !ERROR: DECLARE TARGET directive without arguments or clauses must appear in a subroutine or function
  !$omp declare target
end submodule

! Valid: bare form inside a separate module subprogram (MODULE PROCEDURE)
module mod_with_proc
  interface
    module subroutine mod_sub()
    end subroutine
  end interface
end module

submodule (mod_with_proc) mod_with_proc_impl
contains
  module procedure mod_sub
    !$omp declare target
  end procedure
end submodule

! Valid: bare form inside an interface body in a module
module mod_with_interface
  interface
    subroutine interface_sub()
      !$omp declare target
    end subroutine
  end interface
end module

! Invalid: bare form inside a BLOCK construct (even within a subroutine)
subroutine sub_with_block()
  implicit none
  integer :: outer
  !$omp declare target
  block
    integer :: inner
    !ERROR: DECLARE TARGET directive without arguments or clauses must appear in a subroutine or function
    !$omp declare target
  end block
  outer = 1
end subroutine

! Valid: empty BLOCK in subroutine does not cause false positive
subroutine sub_empty_block()
  implicit none
  integer :: x
  !$omp declare target
  block
  end block
  x = 1
end subroutine
