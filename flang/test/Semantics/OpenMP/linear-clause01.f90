! REQUIRES: openmp_runtime
! RUN: %python %S/../test_errors.py %s %flang_fc1 %openmp_flags
! OpenMP Version 5.2
! Various checks for the linear clause
! 5.4.6 `linear` Clause

! Case 1
subroutine linear_clause_01(arg)
    integer, intent(in) :: arg(:)
    !ERROR: A modifier may not be specified in a LINEAR clause on the DO directive
    !$omp do linear(uval(arg))
    do i = 1, 5
        print *, arg(i)
    end do
end subroutine linear_clause_01

! Case 2
subroutine linear_clause_02(arg_01, arg_02)
    !ERROR: The list item 'arg_01' specified without the REF 'linear-modifier' must be of INTEGER type
    !$omp declare simd linear(val(arg_01))
    real, intent(in) :: arg_01(:)

    !ERROR: The list item 'arg_02' specified without the REF 'linear-modifier' must be of INTEGER type
    !ERROR: If the `linear-modifier` is REF or UVAL, the list item 'arg_02' must be a dummy argument without the VALUE attribute
    !$omp declare simd linear(uval(arg_02))
    !ERROR: The type of 'arg_02' has already been implicitly declared
    integer, value, intent(in) :: arg_02

    !ERROR: The list item 'var' specified without the REF 'linear-modifier' must be of INTEGER type
    !ERROR: If the `linear-modifier` is REF or UVAL, the list item 'var' must be a dummy argument without the VALUE attribute
    !ERROR: The list item `var` must be a dummy argument
    !ERROR: The list item `var` in a LINEAR clause must not be Cray Pointer or a variable with POINTER attribute
    !$omp declare simd linear(uval(var))
    !ERROR: The type of 'var' has already been implicitly declared
    integer, pointer :: var
end subroutine linear_clause_02

! Case 3
subroutine linear_clause_03(arg)
    integer, intent(in) :: arg
    !ERROR: The list item `arg` specified with the REF 'linear-modifier' must be polymorphic variable, assumed-shape array, or a variable with the `ALLOCATABLE` attribute
    !ERROR: List item 'arg' present at multiple LINEAR clauses
    !ERROR: 'arg' appears in more than one data-sharing clause on the same OpenMP directive
    !$omp declare simd linear(ref(arg)) linear(arg)

    integer :: i
    common /cc/ i
    !ERROR: The list item `i` must be a dummy argument
    !ERROR: 'i' is a common block name and must not appear in an LINEAR clause
    !$omp declare simd linear(i)
end subroutine linear_clause_03
