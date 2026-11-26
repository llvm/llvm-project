!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=45

!Check that we still enforce syntactic rules on directives that are not
!present in the specified version.

subroutine interop_no_action(d)
  integer :: d
!ERROR: One of DESTROY, INIT or USE clauses is required on INTEROP directive
!WARNING: DEVICE clause is not allowed on INTEROP directive in OpenMP v4.5, try -fopenmp-version=51 [-Wopenmp-future]
  !$omp interop device(d)
end subroutine

subroutine declare_variant_no_match
!ERROR: MATCH clause is required on DECLARE VARIANT directive
  !$omp declare variant (declare_variant_no_match:vsub)
contains
  subroutine vsub
  end subroutine
end subroutine

subroutine declare_mapper_no_map
  type t
    integer :: x
  end type
!ERROR: MAP clause is required on DECLARE MAPPER directive
  !$omp declare mapper(t :: v)
end subroutine

subroutine scan_no_clause(a, n)
  integer :: n, i, x
  integer :: a(n)
  x = 0
  !$omp do
  do i = 1, n
    x = x + a(i)
!ERROR: One of EXCLUSIVE or INCLUSIVE clauses is required on SCAN directive
    !$omp scan
    a(i) = x
  end do
end subroutine

