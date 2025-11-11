! RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=51

subroutine defaultmap_all_none_no_errors
    implicit none
    real :: array(10)
    integer,  pointer :: ptr(:)
    real, allocatable :: alloca
    integer :: index

    !$omp target defaultmap(none) map(to: index, alloca) map(tofrom: array, ptr)
        do index = 1, 10
            ptr(index) = array(index) + alloca
        end do
    !$omp end target
end subroutine defaultmap_all_none_no_errors

subroutine defaultmap_all_none
    implicit none
    real :: array(10)
    integer,  pointer :: ptr(:)
    real, allocatable :: alloca
    integer :: index
    !$omp target defaultmap(none)
!ERROR: The DEFAULTMAP(NONE) clause requires that 'index' must be listed in a data-sharing attribute, data-mapping attribute, or is_device_ptr clause
        do index = 1, 10
!ERROR: The DEFAULTMAP(NONE) clause requires that 'ptr' must be listed in a data-sharing attribute, data-mapping attribute, or is_device_ptr clause
!ERROR: The DEFAULTMAP(NONE) clause requires that 'index' must be listed in a data-sharing attribute, data-mapping attribute, or is_device_ptr clause
!ERROR: The DEFAULTMAP(NONE) clause requires that 'array' must be listed in a data-sharing attribute, data-mapping attribute, or is_device_ptr clause
!ERROR: The DEFAULTMAP(NONE) clause requires that 'index' must be listed in a data-sharing attribute, data-mapping attribute, or is_device_ptr clause
!ERROR: The DEFAULTMAP(NONE) clause requires that 'alloca' must be listed in a data-sharing attribute, data-mapping attribute, or is_device_ptr clause
            ptr(index) = array(index) + alloca
        end do
    !$omp end target
end subroutine defaultmap_all_none

subroutine defaultmap_scalar_none
    implicit none
    real :: array(10)
    integer,  pointer :: ptr(:)
    real, allocatable :: alloca
    integer :: index

    !$omp target defaultmap(none: scalar)
!ERROR: The DEFAULTMAP(NONE) clause requires that 'index' must be listed in a data-sharing attribute, data-mapping attribute, or is_device_ptr clause
        do index = 1, 10
!ERROR: The DEFAULTMAP(NONE) clause requires that 'index' must be listed in a data-sharing attribute, data-mapping attribute, or is_device_ptr clause
!ERROR: The DEFAULTMAP(NONE) clause requires that 'index' must be listed in a data-sharing attribute, data-mapping attribute, or is_device_ptr clause
            ptr(index) = array(index) + alloca
        end do
    !$omp end target
end subroutine defaultmap_scalar_none

subroutine defaultmap_pointer_none
    implicit none
    real :: array(10)
    integer,  pointer :: ptr(:)
    real, allocatable :: alloca
    integer :: index

    !$omp target defaultmap(none: pointer)
        do index = 1, 10
!ERROR: The DEFAULTMAP(NONE) clause requires that 'ptr' must be listed in a data-sharing attribute, data-mapping attribute, or is_device_ptr clause
            ptr(index) = array(index) + alloca
        end do
    !$omp end target
end subroutine defaultmap_pointer_none

subroutine defaultmap_allocatable_none
    implicit none
    real :: array(10)
    integer,  pointer :: ptr(:)
    real, allocatable :: alloca
    integer :: index

    !$omp target defaultmap(none: allocatable)
        do index = 1, 10
!ERROR: The DEFAULTMAP(NONE) clause requires that 'alloca' must be listed in a data-sharing attribute, data-mapping attribute, or is_device_ptr clause
            ptr(index) = array(index) + alloca
        end do
    !$omp end target
end subroutine defaultmap_allocatable_none

subroutine defaultmap_aggregate_none
    implicit none
    real :: array(10)
    integer,  pointer :: ptr(:)
    real, allocatable :: alloca
    integer :: index

    !$omp target defaultmap(none: aggregate)
        do index = 1, 10
!ERROR: The DEFAULTMAP(NONE) clause requires that 'array' must be listed in a data-sharing attribute, data-mapping attribute, or is_device_ptr clause
            ptr(index) = array(index) + alloca
        end do
    !$omp end target
end subroutine defaultmap_aggregate_none
