! RUN: %python %S/../test_errors.py %s %flang -fopenacc

! Same-kind data-sharing duplicates on an OpenACC directive (e.g.
! private(x, x), private(x) private(x), copyin(x, x) ...) are not errors:
! resolve-directives warns and rewrite-parse-tree drops the duplicate
! occurrences from the clause object lists. Cross-kind duplicates
! (e.g. private(x) firstprivate(x)) and reduction duplicates remain
! hard errors.

program test_dataclause_dedup
  implicit none
  integer :: x, y, z, i

  ! passThis1.f90 pattern: duplicate within a single PRIVATE clause.
  !WARNING: 'x' appears more than once in the same kind of data-sharing clause on an OpenACC directive; duplicate ignored [-Wopenacc-usage]
  !$acc parallel loop private(x, x)
  do i = 1, 10
  end do

  ! passThis2.f90 pattern: duplicate within a single PRIVATE clause across
  ! a continuation, with another variable in between.
  !$acc parallel loop private(x, &
  !WARNING: 'x' appears more than once in the same kind of data-sharing clause on an OpenACC directive; duplicate ignored [-Wopenacc-usage]
  !$acc&               y, x)
  do i = 1, 10
  end do

  ! passThis3.f90 pattern: duplicate across two separate PRIVATE clauses
  ! on the same directive.
  !$acc parallel loop private(x) &
  !WARNING: 'x' appears more than once in the same kind of data-sharing clause on an OpenACC directive; duplicate ignored [-Wopenacc-usage]
  !$acc&              private(y, x)
  do i = 1, 10
  end do

  ! Same patterns generalize to FIRSTPRIVATE.
  !WARNING: 'x' appears more than once in the same kind of data-sharing clause on an OpenACC directive; duplicate ignored [-Wopenacc-usage]
  !$acc parallel loop firstprivate(x, x)
  do i = 1, 10
  end do

  !WARNING: 'x' appears more than once in the same kind of data-sharing clause on an OpenACC directive; duplicate ignored [-Wopenacc-usage]
  !$acc parallel loop firstprivate(x) firstprivate(y, x)
  do i = 1, 10
  end do

  ! Multiple distinct duplicates on a single directive.
  !WARNING: 'x' appears more than once in the same kind of data-sharing clause on an OpenACC directive; duplicate ignored [-Wopenacc-usage]
  !WARNING: 'y' appears more than once in the same kind of data-sharing clause on an OpenACC directive; duplicate ignored [-Wopenacc-usage]
  !$acc parallel loop private(x, y, x, y)
  do i = 1, 10
  end do

  ! Triple occurrence: two duplicates, both warned, only one survives dedup.
  !WARNING: 'x' appears more than once in the same kind of data-sharing clause on an OpenACC directive; duplicate ignored [-Wopenacc-usage]
  !WARNING: 'x' appears more than once in the same kind of data-sharing clause on an OpenACC directive; duplicate ignored [-Wopenacc-usage]
  !$acc parallel loop private(x, x, x)
  do i = 1, 10
  end do

  ! Cross-kind duplicates on the same directive remain hard errors.
  !ERROR: 'x' appears in more than one data-sharing clause on the same OpenACC directive
  !$acc parallel loop private(x) firstprivate(x)
  do i = 1, 10
  end do

  ! Reduction is excluded from the benign case: same-flag duplicates may
  ! differ in operator, which is a real conflict.
  !ERROR: 'x' appears in more than one data-sharing clause on the same OpenACC directive
  !$acc parallel loop reduction(+:x) reduction(*:x)
  do i = 1, 10
  end do

  ! Regression coverage for non-bare designators: the dedup machinery only
  ! examines simple-Name DataRefs, so distinct array elements and array
  ! sections must pass through untouched, with no warning and no erasure.
  block
    integer :: arr(10)
    integer, target :: t1, t2
    integer, pointer :: p
    type :: pt
      integer :: a
      integer :: b
    end type
    type(pt) :: s

    ! Different array elements -- not duplicates.
    !$acc parallel loop private(arr(1), arr(2))
    do i = 1, 10
    end do

    ! Different array sections -- not duplicates.
    !$acc parallel loop private(arr(1:5), arr(6:10))
    do i = 1, 10
    end do

    ! Same array element listed twice -- not deduped, since GetDesignatorName-
    ! IfDataRef returns null for ArrayElement and CheckMultipleAppearances
    ! is never invoked. Compiles without diagnostics.
    !$acc parallel loop private(arr(1), arr(1))
    do i = 1, 10
    end do

    ! Same array section listed twice -- same reasoning, no diagnostic.
    !$acc parallel loop private(arr(1:5), arr(1:5))
    do i = 1, 10
    end do

    ! Distinct structure components -- not duplicates.
    !$acc parallel loop private(s%a, s%b)
    do i = 1, 10
    end do

    ! Mixing a bare-name designator and an array-element designator on the
    ! same symbol must not trigger dedup -- the array element doesn't go
    ! through the duplicate check at all.
    !$acc parallel loop private(arr, arr(1))
    do i = 1, 10
    end do
  end block

end program
