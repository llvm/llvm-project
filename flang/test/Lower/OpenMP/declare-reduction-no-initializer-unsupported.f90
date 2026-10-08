! RUN: split-file %s %t
! RUN: not %flang_fc1 -emit-hlfir -fopenmp %t/dyn_char.f90 -o - 2>&1 | FileCheck -check-prefix=DYN-CHAR %s
! RUN: not %flang_fc1 -emit-hlfir -fopenmp %t/non_trivial.f90 -o - 2>&1 | FileCheck -check-prefix=NON-TRIVIAL %s

! Test that unsupported types in declare reduction without initializer
! produce a TODO diagnostic rather than crashing.

! DYN-CHAR: not yet implemented: declare reduction currently only supports trivial types, fixed-length CHARACTER, or derived types containing them
! NON-TRIVIAL: not yet implemented: declare reduction currently only supports trivial types, fixed-length CHARACTER, or derived types containing them

!--- dyn_char.f90
subroutine dyn_char(s, n)
  integer, intent(in) :: n
  character(len=n) :: s
  integer :: i

  !$omp declare reduction(char_max: character(len=n): omp_out = max(omp_out, omp_in))

  !$omp parallel do reduction(char_max: s)
  do i = 1, 4
    s = s
  end do
end subroutine

!--- non_trivial.f90
subroutine non_trivial(x)
  type :: alloc_type
    integer, allocatable :: data(:)
  end type

  type(alloc_type) :: x
  integer :: i

  !$omp declare reduction(my_add: alloc_type: omp_out%data = omp_out%data + omp_in%data)

  !$omp parallel do reduction(my_add: x)
  do i = 1, 4
    x%data(i) = x%data(i) + 1
  end do
end subroutine
