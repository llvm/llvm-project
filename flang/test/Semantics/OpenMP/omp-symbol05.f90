! RUN: %python %S/../test_symbols.py %s %flang_fc1 -fopenmp

! 2.15.2 threadprivate Directive
! The threadprivate directive specifies that variables are replicated,
! with each thread having its own copy. When threadprivate variables are
! referenced in the OpenMP region, we know they are already private to
! their threads, so no new symbol needs to be created.

!DEF: /mm Module
module mm
  !$omp threadprivate (i)
contains
  !DEF: /mm/foo PUBLIC (Subroutine) Subprogram
  subroutine foo
    !DEF: /mm/foo/a ObjectEntity INTEGER(4)
    integer :: a = 3
    !$omp parallel
    !REF: /mm/foo/a
    a = 1
    !DEF: /mm/i PUBLIC (Implicit, OmpThreadprivate) ObjectEntity INTEGER(4)
    !REF: /mm/foo/a
    i = a
    !$omp end parallel
    !REF: /mm/foo/a
    print *, a
    block
      !DEF: /mm/foo/Block2/i ObjectEntity REAL(4)
      real i
      !REF: /mm/foo/Block2/i
      i = 3.14
    end block
  end subroutine foo
end module mm
!DEF: /tt MainProgram
program tt
  !REF: /mm
  use :: mm
  !DEF: /tt/foo (Subroutine) Use
  call foo
end program tt
