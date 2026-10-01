! RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=60

! OpenMP 6.0, 9.8 declare_simd Directive, Fortran restriction:
! "Any declare_simd directive must appear in the specification part of a
! subroutine subprogram, function subprogram, or interface body to which it
! applies."

module m
  procedure() :: ext

!ERROR: DECLARE_SIMD directive must appear in the specification part of a subroutine subprogram, function subprogram, or interface body
  !$omp declare simd

!ERROR: DECLARE_SIMD directive must appear in the specification part of a subroutine subprogram, function subprogram, or interface body
  !$omp declare_simd(ext)

contains

  ! Ok: module subprogram.
  subroutine mod_sub(i)
    integer :: i
    !$omp declare simd(mod_sub) linear(i:1)
    i = i + 1
  end subroutine

  ! Ok: module function.
  integer function mod_fun(i)
    integer :: i
    !$omp declare_simd
    mod_fun = i
  end function
end module

submodule (m) sm
!ERROR: DECLARE_SIMD directive must appear in the specification part of a subroutine subprogram, function subprogram, or interface body
  !$omp declare simd
end submodule

block data bd
  integer :: x
  common /blk/ x
!ERROR: DECLARE_SIMD directive must appear in the specification part of a subroutine subprogram, function subprogram, or interface body
  !$omp declare simd
  data x /0/
end block data

program main
!ERROR: DECLARE_SIMD directive must appear in the specification part of a subroutine subprogram, function subprogram, or interface body
  !$omp declare simd
end program

! The specification part of a BLOCK construct is not the specification part of
! the enclosing subprogram.
subroutine in_block(i)
  integer :: i
  block
!ERROR: DECLARE_SIMD directive must appear in the specification part of a subroutine subprogram, function subprogram, or interface body
    !$omp declare simd
    i = i + 1
  end block
end subroutine

! Ok: external subroutine.
subroutine ext_sub(i)
  integer :: i
  !$omp declare simd(ext_sub) linear(i:1)
  i = i + 1
end subroutine

! Ok: external function, with an internal subprogram.
integer function ext_fun(i)
  integer :: i
  !$omp declare simd(ext_fun)
  ext_fun = inner(i)
contains
  ! Ok: internal subprogram.
  integer function inner(j)
    integer :: j
    !$omp declare simd(inner) linear(j:1)
    inner = j
  end function
end function

! Ok: interface body (the directive is ignored, hence the warning).
subroutine has_interface
  interface
    subroutine iface_sub(i)
!WARNING: 'DECLARE SIMD' directive in an interface body has no effect [-Wopenmp-usage]
      !$omp declare simd(iface_sub) linear(i:1)
      integer :: i
    end subroutine
  end interface
end subroutine
