! RUN: %python %S/../test_errors.py %s %flang -fopenacc

! Same-kind data-sharing duplicates on an OpenACC directive (e.g.
! private(x, x), private(x) private(x), copyin(x, x) ...) are not errors:
! resolve-directives warns and rewrite-parse-tree drops the duplicate
! occurrences from the clause object lists. When one same-kind object contains
! another, the contained occurrence is likewise dropped. Cross-kind
! duplicates (e.g. private(x) firstprivate(x)), partial overlaps, disjoint
! parts of one array, and reduction duplicates remain hard errors.

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

  ! A reduction still conflicts with explicit privatization on the same
  ! directive.
  !ERROR: 'x' appears in more than one data-sharing clause on the same OpenACC directive
  !$acc parallel private(x) reduction(+:x)
  x = x + 1
  !$acc end parallel

  !ERROR: 'x' appears in more than one data-sharing clause on the same OpenACC directive
  !$acc serial firstprivate(x) reduction(+:x)
  x = x + 1
  !$acc end serial

  !ERROR: 'x' appears in more than one data-sharing clause on the same OpenACC directive
  !$acc parallel loop private(x) reduction(+:x)
  do i = 1, 10
    x = x + i
  end do

  !ERROR: 'x' appears in more than one data-sharing clause on the same OpenACC directive
  !$acc serial loop firstprivate(x) reduction(+:x)
  do i = 1, 10
    x = x + i
  end do

  ! Reduction is excluded from the benign case: same-flag duplicates may
  ! differ in operator, which is a real conflict.
  !ERROR: 'x' appears in more than one data-sharing clause on the same OpenACC directive
  !$acc parallel loop reduction(+:x) reduction(*:x)
  do i = 1, 10
  end do

  ! Regression coverage for non-bare designators: distinct array elements and
  ! sections must pass through untouched, while exact duplicates are diagnosed
  ! precisely rather than by the base array symbol.
  block
    integer :: arr(10)
    integer :: lo, mid, hi, idx
    integer, parameter :: left = 1, split = 5, right = 10
    integer, target :: t1, t2
    integer, pointer :: p
    type :: pt
      integer :: a
      integer :: b
    end type
    type(pt) :: s

    ! Distinct elements of one array still represent the same data-sharing
    ! entity and cannot be lowered as separate private operands.
    !ERROR: 'arr(2)' is a different part of an object that already appears in the same kind of data-sharing clause on the same OpenACC directive
    !$acc parallel loop private(arr(1), arr(2))
    do i = 1, 10
    end do

    !ERROR: 'arr(6:10)' is a different part of an object that already appears in the same kind of data-sharing clause on the same OpenACC directive
    !$acc parallel loop private(arr(1:5), arr(6:10))
    do i = 1, 10
    end do

    ! Different array elements in different data-sharing clauses conflict.
    !ERROR: 'arr(2)' appears in more than one data-sharing clause on the same OpenACC directive
    !$acc parallel loop private(arr(1)) firstprivate(arr(2))
    do i = 1, 10
    end do

    ! Reversing the clause order does not change the entity-level conflict.
    !ERROR: 'arr(1)' appears in more than one data-sharing clause on the same OpenACC directive
    !$acc parallel loop firstprivate(arr(2)) private(arr(1))
    do i = 1, 10
    end do

    !ERROR: 'arr(6:10)' appears in more than one data-sharing clause on the same OpenACC directive
    !$acc parallel loop private(arr(1:5)) firstprivate(arr(6:10))
    do i = 1, 10
    end do

    !ERROR: 'arr(1:5)' appears in more than one data-sharing clause on the same OpenACC directive
    !$acc parallel loop firstprivate(arr(6:10)) private(arr(1:5))
    do i = 1, 10
    end do

    ! A literal element outside a literal section is disjoint across
    ! data-sharing kinds but still belongs to the same array entity.
    !ERROR: 'arr(6)' appears in more than one data-sharing clause on the same OpenACC directive
    !$acc parallel loop private(arr(1:5)) firstprivate(arr(6))
    do i = 1, 10
    end do

    ! Named constant sections that fold to disjoint ranges still conflict.
    !ERROR: 'arr(split+1:right)' appears in more than one data-sharing clause on the same OpenACC directive
    !$acc parallel loop private(arr(left:split)) firstprivate(arr(split+1:right))
    do i = 1, 10
    end do

    ! Same array element listed twice in the same data-sharing clause.
    !WARNING: 'arr(1)' appears more than once in the same kind of data-sharing clause on an OpenACC directive; duplicate ignored [-Wopenacc-usage]
    !$acc parallel loop private(arr(1), arr(1))
    do i = 1, 10
    end do

    ! Same array element with different source spelling.
    !WARNING: 'arr(01)' appears more than once in the same kind of data-sharing clause on an OpenACC directive; duplicate ignored [-Wopenacc-usage]
    !$acc parallel loop private(arr(1), arr(01))
    do i = 1, 10
    end do

    ! Same array element listed in conflicting data-sharing clauses.
    !ERROR: 'arr(1)' appears in more than one data-sharing clause on the same OpenACC directive
    !$acc parallel loop private(arr(1)) firstprivate(arr(1))
    do i = 1, 10
    end do

    ! Same array section listed twice in the same data-sharing clause.
    !WARNING: 'arr(1:5)' appears more than once in the same kind of data-sharing clause on an OpenACC directive; duplicate ignored [-Wopenacc-usage]
    !$acc parallel loop private(arr(1:5), arr(1:5))
    do i = 1, 10
    end do

    ! Same array section listed in conflicting data-sharing clauses.
    !ERROR: 'arr(1:5)' appears in more than one data-sharing clause on the same OpenACC directive
    !$acc parallel loop private(arr(1:5)) firstprivate(arr(1:5))
    do i = 1, 10
    end do

    ! Same array section with different source spelling.
    !WARNING: 'arr(01:05)' appears more than once in the same kind of data-sharing clause on an OpenACC directive; duplicate ignored [-Wopenacc-usage]
    !$acc parallel loop private(arr(1:5), arr(01:05))
    do i = 1, 10
    end do

    ! Equivalent array sections with different source spelling in conflicting
    ! data-sharing clauses.
    !ERROR: 'arr(01:05)' appears in more than one data-sharing clause on the same OpenACC directive
    !$acc parallel loop private(arr(1:5)) firstprivate(arr(01:05))
    do i = 1, 10
    end do

    ! Non-identical sections that overlap in the same data-sharing kind are
    ! rejected until precise overlap support is implemented.
    !ERROR: 'arr(5:10)' overlaps another object in the same kind of data-sharing clause on the same OpenACC directive
    !$acc parallel loop private(arr(1:5), arr(5:10))
    do i = 1, 10
    end do

    ! Overlapping literal sections in different data-sharing kinds conflict.
    !ERROR: 'arr(5:10)' appears in more than one data-sharing clause on the same OpenACC directive
    !$acc parallel loop private(arr(1:5)) firstprivate(arr(5:10))
    do i = 1, 10
    end do

    ! A contained object is redundant in the same data-sharing kind. Keep the
    ! containing object and ignore the contained occurrence in either order.
    !WARNING: 'arr(3)' is contained in another object in the same kind of data-sharing clause on an OpenACC directive; contained object ignored [-Wopenacc-usage]
    !$acc parallel loop private(arr(1:5), arr(3))
    do i = 1, 10
    end do

    !WARNING: 'arr(3)' is contained in another object in the same kind of data-sharing clause on an OpenACC directive; contained object ignored [-Wopenacc-usage]
    !$acc parallel loop private(arr(3), arr(1:5))
    do i = 1, 10
    end do

    ! An element contained in a section conflicts across data-sharing kinds.
    !ERROR: 'arr(3)' appears in more than one data-sharing clause on the same OpenACC directive
    !$acc parallel loop private(arr(1:5)) firstprivate(arr(3))
    do i = 1, 10
    end do

    ! Variable index/section containment cannot be proven, so separate
    ! appearances of the same array entity are rejected.
    !ERROR: 'arr(lo:hi)' is a different part of an object that already appears in the same kind of data-sharing clause on the same OpenACC directive
    !$acc parallel loop private(arr(idx), arr(lo:hi))
    do i = 1, 10
    end do

    ! Variable index/section overlap is ambiguous across data-sharing kinds.
    !ERROR: 'arr(idx)' appears in more than one data-sharing clause on the same OpenACC directive
    !$acc parallel loop private(arr(lo:hi)) firstprivate(arr(idx))
    do i = 1, 10
    end do

    ! Identical variable indices conflict across data-sharing kinds.
    !ERROR: 'arr(idx)' appears in more than one data-sharing clause on the same OpenACC directive
    !$acc parallel loop private(arr(idx)) firstprivate(arr(idx))
    do i = 1, 10
    end do

    ! Variable section overlap cannot be proven, so separate appearances of
    ! the same array entity are rejected.
    !ERROR: 'arr(mid:hi)' is a different part of an object that already appears in the same kind of data-sharing clause on the same OpenACC directive
    !$acc parallel loop private(arr(lo:hi), arr(mid:hi))
    do i = 1, 10
    end do

    ! Variable section overlap is ambiguous across data-sharing kinds.
    !ERROR: 'arr(mid:hi)' appears in more than one data-sharing clause on the same OpenACC directive
    !$acc parallel loop private(arr(lo:hi)) firstprivate(arr(mid:hi))
    do i = 1, 10
    end do

    !ERROR: 'arr(lo:hi)' appears in more than one data-sharing clause on the same OpenACC directive
    !$acc parallel loop firstprivate(arr(mid:hi)) private(arr(lo:hi))
    do i = 1, 10
    end do

    ! Identical variable sections conflict across data-sharing kinds.
    !ERROR: 'arr(lo:hi)' appears in more than one data-sharing clause on the same OpenACC directive
    !$acc parallel loop private(arr(lo:hi)) firstprivate(arr(lo:hi))
    do i = 1, 10
    end do

    ! Distinct structure components -- not duplicates.
    !$acc parallel loop private(s%a, s%b)
    do i = 1, 10
    end do

    ! A whole array contains an element, so only the whole-array occurrence is
    ! retained.
    !WARNING: 'arr(1)' is contained in another object in the same kind of data-sharing clause on an OpenACC directive; contained object ignored [-Wopenacc-usage]
    !$acc parallel loop private(arr, arr(1))
    do i = 1, 10
    end do

    ! Eugene's whole-array/proper-subset example, in both source orders.
    !WARNING: 'arr(1:5)' is contained in another object in the same kind of data-sharing clause on an OpenACC directive; contained object ignored [-Wopenacc-usage]
    !$acc parallel private(arr) private(arr(1:5))
    arr(9) = 1
    !$acc end parallel

    !WARNING: 'arr(1:5)' is contained in another object in the same kind of data-sharing clause on an OpenACC directive; contained object ignored [-Wopenacc-usage]
    !$acc parallel private(arr(1:5)) private(arr)
    arr(9) = 1
    !$acc end parallel

    ! All contained occurrences are removed before the remaining paths are
    ! compared, so unknown selectors behave uniformly in every source order.
    !WARNING: 'arr(idx)' is contained in another object in the same kind of data-sharing clause on an OpenACC directive; contained object ignored [-Wopenacc-usage]
    !WARNING: 'arr(mid)' is contained in another object in the same kind of data-sharing clause on an OpenACC directive; contained object ignored [-Wopenacc-usage]
    !$acc parallel private(arr(idx), arr(mid)) private(arr)
    arr(9) = 1
    !$acc end parallel

    !WARNING: 'arr(idx)' is contained in another object in the same kind of data-sharing clause on an OpenACC directive; contained object ignored [-Wopenacc-usage]
    !WARNING: 'arr(mid)' is contained in another object in the same kind of data-sharing clause on an OpenACC directive; contained object ignored [-Wopenacc-usage]
    !$acc parallel private(arr(idx)) private(arr) private(arr(mid))
    arr(9) = 1
    !$acc end parallel

    !WARNING: 'arr(idx)' is contained in another object in the same kind of data-sharing clause on an OpenACC directive; contained object ignored [-Wopenacc-usage]
    !WARNING: 'arr(mid)' is contained in another object in the same kind of data-sharing clause on an OpenACC directive; contained object ignored [-Wopenacc-usage]
    !$acc parallel private(arr) private(arr(idx), arr(mid))
    arr(9) = 1
    !$acc end parallel
  end block

end program
