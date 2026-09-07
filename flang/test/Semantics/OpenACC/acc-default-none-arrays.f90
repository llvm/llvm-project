! RUN: %python %S/../test_errors.py %s %flang -fopenacc -fno-openacc-default-none-scalars-strict -Wno-openacc-default-none-scalars-strict

! Verify that array sections explicitly listed in OpenACC data clauses are
! correctly registered as having a DSA, so DEFAULT(NONE) does not produce
! spurious errors.  This does not implement section-level overlap
! detection or deduplication; duplicate/conflict diagnostics only apply to bare
! names.  This also covers the substring-in-clause error.

! 1. Data-mapping clauses with array sections: no DEFAULT(NONE) errors.
subroutine test_data_mapping_sections(n)
  implicit none
  integer, intent(in) :: n
  real :: a(n*n), b(n), c(n)
  integer :: i, j
  real :: temp
  !$acc kernels default(none) copyin(a(1:n*n), b(1:n)) copy(c(1:n))
  !$acc loop gang private(temp)
  do i = 1, n
    temp = 0.0
    !$acc loop vector reduction(+:temp)
    do j = 1, n
      temp = temp + a((i-1)*n+j) * b(j)
    enddo
    c(i) = temp
  enddo
  !$acc end kernels
end subroutine

! 2. Private clause with array section: no DEFAULT(NONE) error.
subroutine test_private_section(n)
  implicit none
  integer, intent(in) :: n
  real :: a(n)
  integer :: i
  !$acc parallel loop default(none) private(a(:))
  do i = 1, n
    a(i) = 0.0
  end do
  !$acc end parallel loop
end subroutine

! 3. Parallel with copyin array section and separate parallel body: no error.
subroutine test_parallel_copyin_section(n)
  implicit none
  integer, intent(in) :: n
  real :: x(n), y(n)
  integer :: i
  !$acc parallel loop default(none) copyin(x(1:n)) copyout(y(1:n))
  do i = 1, n
    y(i) = x(i) * 2.0
  end do
  !$acc end parallel loop
end subroutine

! 4. Unlisted array still errors under DEFAULT(NONE) (regression check).
subroutine test_unlisted_array(n)
  implicit none
  integer, intent(in) :: n
  real :: a(n), b(n)
  integer :: i
  !$acc parallel default(none) copyin(a(1:n))
  !ERROR: The DEFAULT(NONE) clause requires that 'b' must be listed in a data-mapping clause
  b(1) = a(1)
  !$acc end parallel
end subroutine

! 5. Duplicate bare-name under the same data-sharing clause: warn and dedup.
!    (Array sections like private(a(1:5), a(6:10)) are not deduplicated or
!    checked for overlap because base-name comparison cannot distinguish
!    different sections of the same array.)
subroutine test_duplicate_private_bare(n)
  implicit none
  integer, intent(in) :: n
  real :: a(n)
  integer :: i
  !WARNING: 'a' appears more than once in the same kind of data-sharing clause on an OpenACC directive; duplicate ignored [-Wopenacc-usage]
  !$acc parallel loop default(none) private(a, a)
  do i = 1, n
    a(i) = 0.0
  end do
  !$acc end parallel loop
end subroutine

! 6. Same bare-name variable in two different data-sharing clauses: error.
subroutine test_cross_kind_bare(n)
  implicit none
  integer, intent(in) :: n
  real :: a(n)
  integer :: i
  !ERROR: 'a' appears in more than one data-sharing clause on the same OpenACC directive
  !$acc parallel loop default(none) private(a) firstprivate(a)
  do i = 1, n
    a(i) = 0.0
  end do
  !$acc end parallel loop
end subroutine

! 7. Substring in an OpenACC clause is disallowed.
subroutine test_substring()
  implicit none
  character(len=10) :: str
  !ERROR: Substrings are not allowed on OpenACC directives or clauses
  !$acc parallel default(none) copyin(str(1:5))
  !$acc end parallel
end subroutine

! 8. Same array section in conflicting private and copy clauses.
! TODO: cross-kind detection for array sections is not implemented; no error
!       produced for 'a(1:n)' appearing in both copy and private.
subroutine test_cross_kind_sections(n)
  implicit none
  integer, intent(in) :: n
  real :: a(n)
  integer :: i
  !$acc parallel loop default(none) copy(a(1:n)) private(a(1:n))
  do i = 1, n
    a(i) = 0.0
  end do
  !$acc end parallel loop
end subroutine

! 9. Different sections of the same array in conflicting copy and private clauses.
! TODO: cross-kind detection for array sections is not implemented; no error
!       produced for 'a' appearing in both copy and private.
subroutine test_cross_kind_sections2(n)
  implicit none
  integer, intent(in) :: n
  real :: a(n)
  integer :: i
  !$acc parallel loop default(none) copy(a(1:n/2)) private(a(n/2+1:n))
  do i = 1, n
    a(i) = 0.0
  end do
  !$acc end parallel loop
end subroutine
