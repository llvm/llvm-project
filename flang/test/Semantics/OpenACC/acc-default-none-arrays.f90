! RUN: %python %S/../test_errors.py %s %flang -fopenacc -fno-openacc-default-none-scalars-strict -Wno-openacc-default-none-scalars-strict

! Verify that array sections explicitly listed in OpenACC data clauses are
! correctly registered as having a DSA, so DEFAULT(NONE) uses path containment
! rather than treating a listed array section as covering every reference to the
! base array.  This also covers the substring-in-clause error.

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

subroutine test_default_none_literal_section_contains_element()
  implicit none
  real :: a(10)
  !$acc parallel default(none) copy(a(1:5))
  a(3) = 1.0
  !$acc end parallel
end subroutine

subroutine test_default_none_literal_section_rejects_disjoint_element()
  implicit none
  real :: a(10)
  !$acc parallel default(none) copy(a(1:5))
  !ERROR: The DEFAULT(NONE) clause requires that 'a' must be listed in a data-mapping clause
  a(6) = 1.0
  !$acc end parallel
end subroutine

subroutine test_default_none_literal_section_rejects_partially_overlapping_section()
  implicit none
  real :: a(10)
  !$acc parallel default(none) copy(a(1:5))
  !ERROR: The DEFAULT(NONE) clause requires that 'a' must be listed in a data-mapping clause
  a(5:10) = 1.0
  !$acc end parallel
end subroutine

subroutine test_default_none_literal_section_rejects_full_section()
  implicit none
  real :: a(10)
  !$acc parallel default(none) copy(a(1:5))
  !ERROR: The DEFAULT(NONE) clause requires that 'a' must be listed in a data-mapping clause
  a(:) = 1.0
  !$acc end parallel
end subroutine

subroutine test_default_none_full_section_contains_element()
  implicit none
  real :: a(10)
  !$acc parallel default(none) copy(a(:))
  a = 0.0
  a(10) = 1.0
  !$acc end parallel
end subroutine

subroutine test_default_none_variable_section_lenient(n, lo, hi, i, j, mid)
  implicit none
  integer, intent(in) :: n, lo, hi, i, j, mid
  real :: a(n), b(n), c(n)
  !$acc parallel default(none) copy(a(lo:hi), b(i), c(1:5))
  a(i) = 1.0
  a(mid:hi) = 2.0
  b(j) = 3.0
  c(j) = 4.0
  !$acc end parallel
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
!    Exact section duplicates are also diagnosed, but overlapping or distinct
!    sections like private(a(1:5), a(6:10)) are not treated as duplicates.
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

! 7. Data-sharing entries are scoped to one directive.  Reusing an object in a
!    later, sequential region must not be diagnosed as a duplicate.
subroutine test_sequential_regions_have_independent_data_sharing_entries()
  implicit none
  real :: a(10)
  !$acc parallel copy(a)
  a = 1.0
  !$acc end parallel
  !$acc parallel copy(a)
  a = 2.0
  !$acc end parallel
end subroutine

! 8. Substring in an OpenACC clause is disallowed.
subroutine test_substring()
  implicit none
  character(len=10) :: str
  !ERROR: Substrings are not allowed on OpenACC directives or clauses
  !$acc parallel default(none) copyin(str(1:5))
  !$acc end parallel
end subroutine

! 8a. A coindexed object is not an OpenACC data-clause var.
subroutine test_coindexed_object()
  implicit none
  integer, save :: coarray[*]
  !ERROR: Coindexed objects are not allowed on OpenACC directives or clauses
  !$acc parallel default(none) copyin(coarray[1])
  !$acc end parallel
end subroutine

! 8b. Invalid objects must retain their ordinary Fortran semantic errors;
!     they must not be mistaken for a resolved OpenACC object that happens
!     not to have a DesignatorPath.
subroutine test_unresolved_clause_objects()
  implicit none
  type :: t
    integer :: present
  end type
  type(t) :: x
  !ERROR: Component 'missing' not found in derived type 't'
  !$acc parallel copyin(x%missing)
  !$acc end parallel
  !ERROR: No explicit type declared for 'missing_substring'
  !$acc parallel copyin(missing_substring(1:5))
  !$acc end parallel
  !ERROR: No explicit type declared for 'missing_coarray'
  !$acc parallel copyin(missing_coarray[1])
  !$acc end parallel
end subroutine

! 9. The same array section may appear in a data-action clause and a
! data-sharing clause.
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

! 10. Different sections may likewise appear in data-action and data-sharing
! clauses.
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
