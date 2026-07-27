! RUN: %python %S/test_errors.py %s %flang_fc1
program int_array_alloc_03
    implicit none
    real, allocatable, dimension(:) :: rank1_test_array
    real, allocatable, dimension(:,:,:) :: test_array

    integer :: seven = 7
    integer :: valid_lower(3) = [1,1,1]
    integer :: lower(4), upper(4)
    integer :: rank_2_array(3,3), rank_3_array(3,3,3)
    ! Positive test cases, expecting no errors
    ! Test direct use of scalar integer and array integer expressions
    !ERROR: TODO: AllocateShapeBoundsSpec semantic checks in check-allocate.cpp
    allocate(rank1_test_array([5]))
    !ERROR: TODO: AllocateShapeBoundsSpec semantic checks in check-allocate.cpp
    allocate(rank1_test_array([1]:[5]))
    !ERROR: TODO: AllocateShapeBoundsSpec semantic checks in check-allocate.cpp
    allocate(rank1_test_array(1:[5]))
    !ERROR: TODO: AllocateShapeBoundsSpec semantic checks in check-allocate.cpp
    allocate(rank1_test_array([1]:5))

    ! Test indirect use of scalar integer and array integer expressions
    ! array : array
    !ERROR: TODO: AllocateShapeBoundsSpec semantic checks in check-allocate.cpp
    allocate(test_array([1,2,3] : [1,2,3] + 1))
    ! array : array
    !ERROR: TODO: AllocateShapeBoundsSpec semantic checks in check-allocate.cpp
    allocate(test_array(valid_lower - 1 : seven * (valid_lower + seven)))
    ! array : scalar (broadcast)
    !ERROR: TODO: AllocateShapeBoundsSpec semantic checks in check-allocate.cpp
    allocate(test_array(valid_lower : return_seven()))
    ! scalar : array (broadcast)
    !ERROR: TODO: AllocateShapeBoundsSpec semantic checks in check-allocate.cpp
    allocate(test_array(seven : [9,9,9]))

    !Negative test cases, expecting errors
    !ERROR: Must have INTEGER type, but is REAL(4)
    !ERROR: TODO: AllocateShapeBoundsSpec semantic checks in check-allocate.cpp
    allocate(test_array([1.2,2.2,3.2]:[1,2,3]))

    !future_ERROR: ALLOCATE bounds integer rank-1 arrays must have the same size; lower bounds has 3 elements, upper bounds has 2 elements
    !ERROR: TODO: AllocateShapeBoundsSpec semantic checks in check-allocate.cpp
    allocate(test_array([1,2,3]:[3,3]))

    !future_ERROR: ALLOCATE bounds integer rank-1 arrays have 4 elements but allocatable object 'test_array' has rank 3
    !ERROR: TODO: AllocateShapeBoundsSpec semantic checks in check-allocate.cpp
    allocate(test_array(lower:upper))
    !future_ERROR: ALLOCATE upper bounds integer rank-1 array has 4 elements but allocatable object 'test_array' has rank 3
    !ERROR: TODO: AllocateShapeBoundsSpec semantic checks in check-allocate.cpp
    allocate(test_array(7 : [1,2,3,4]))
    !future_ERROR: ALLOCATE lower bounds integer rank-1 array has 2 elements but allocatable object 'test_array' has rank 3
    !ERROR: TODO: AllocateShapeBoundsSpec semantic checks in check-allocate.cpp
    allocate(test_array([1,2] : 7))

    !future_ERROR: Integer array used as upper bounds in ALLOCATE must be rank-1 but is rank-3
    !ERROR: TODO: AllocateShapeBoundsSpec semantic checks in check-allocate.cpp
    allocate(test_array([1,2,4] : rank_3_array))
    !future_ERROR: Integer array used as lower bounds in ALLOCATE must be rank-1 but is rank-2
    !ERROR: TODO: AllocateShapeBoundsSpec semantic checks in check-allocate.cpp
    allocate(test_array(rank_2_array : [1,2,4]))
    !future_ERROR: Integer array used as lower bounds in ALLOCATE must be rank-1 but is rank-2
    !future_ERROR: Integer array used as upper bounds in ALLOCATE must be rank-1 but is rank-3
    !ERROR: TODO: AllocateShapeBoundsSpec semantic checks in check-allocate.cpp
    allocate(test_array(rank_2_array : rank_3_array))
    !future_ERROR: Integer array used as lower bounds in ALLOCATE must be rank-1 but is rank-2
    !ERROR: TODO: AllocateShapeBoundsSpec semantic checks in check-allocate.cpp
    allocate(test_array(rank_2_array : 7))
    !future_ERROR: Integer array used as upper bounds in ALLOCATE must be rank-1 but is rank-3
    !ERROR: TODO: AllocateShapeBoundsSpec semantic checks in check-allocate.cpp
    allocate(test_array(7 : rank_3_array))

    ! Test that any comma list is parsed as AllocateShapeSpecList and not rewritten 
    ! to AllocateShapeSpecArray, giving error messages expecting same number of 
    ! aruments as rank of test_array and scalar integers
    !ERROR: The number of shape specifications, when they appear, must match the rank of allocatable object
    !ERROR: Must be a scalar value, but is a rank-1 array
    !ERROR: Must be a scalar value, but is a rank-1 array
    !ERROR: Must be a scalar value, but is a rank-1 array
    !ERROR: Must have INTEGER type, but is REAL(4)
    allocate(test_array([1,2,3] : [2,3,4], 3, [1,2,3], 5.2))

  contains
    subroutine tmp02(unknown_size, test_ptr_01)
        real, allocatable, dimension(:,:,:), INTENT(OUT) :: test_ptr_01
        integer, INTENT(IN) :: unknown_size
        integer :: lower(unknown_size), upper(unknown_size)

        !future_ERROR: Rank-1 integer array used as upper bounds in ALLOCATE must have constant size
        !ERROR: TODO: AllocateShapeBoundsSpec semantic checks in check-allocate.cpp
        allocate(test_ptr_01(upper))
        !future_ERROR: Rank-1 integer array used as lower bounds in ALLOCATE must have constant size
        !future_ERROR: Rank-1 integer array used as upper bounds in ALLOCATE must have constant size
        !ERROR: TODO: AllocateShapeBoundsSpec semantic checks in check-allocate.cpp
        allocate(test_ptr_01(lower : upper))
        !future_ERROR: Rank-1 integer array used as lower bounds in ALLOCATE must have constant size
        !ERROR: TODO: AllocateShapeBoundsSpec semantic checks in check-allocate.cpp
        allocate(test_ptr_01(lower : 10))
    end subroutine

    function return_seven() 
        integer :: return_seven
        return_seven = 7
    end function 

end program
