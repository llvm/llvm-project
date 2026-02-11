! RUN: %python %S/test_errors.py %s %flang_fc1
program int_array_alloc_03
    implicit none
    real, allocatable, dimension(:,:,:) :: test_array
    
    integer :: lower(4), upper(4)
    integer :: rank_2_array(3,3), rank_3_array(3,3,3)
    
    !ERROR: Must have INTEGER type, but is REAL(4)
    allocate( test_array([1.2,2.2,3.2]:[1,2,3]))

    !ERROR: ALLOCATE bounds integer rank-1 arrays must have the same size; lower bounds has 3 elements, upper bounds has 2 elements
    allocate( test_array([1,2,3]:[3,3]))

    !ERROR: ALLOCATE bounds integer rank-1 arrays have 4 elements but allocatable object 'test_array' has rank 3
    allocate( test_array(lower:upper) )
    !ERROR: ALLOCATE upper bounds integer rank-1 array has 4 elements but allocatable object 'test_array' has rank 3
    allocate( test_array(7 : [1,2,3,4]))
    !ERROR: ALLOCATE lower bounds integer rank-1 array has 2 elements but allocatable object 'test_array' has rank 3
    allocate( test_array([1,2] : 7))

    !ERROR: Integer array used as upper bounds in ALLOCATE must be rank-1 but is rank-3
    allocate( test_array([1,2,4] : rank_3_array))
    !ERROR: Integer array used as lower bounds in ALLOCATE must be rank-1 but is rank-2
    allocate( test_array(rank_2_array : [1,2,4]))
    !ERROR: Integer array used as lower bounds in ALLOCATE must be rank-1 but is rank-2
    !ERROR: Integer array used as upper bounds in ALLOCATE must be rank-1 but is rank-3
    allocate( test_array(rank_2_array : rank_3_array))
    !ERROR: Integer array used as lower bounds in ALLOCATE must be rank-1 but is rank-2
    allocate( test_array(rank_2_array : 7))
    !ERROR: Integer array used as upper bounds in ALLOCATE must be rank-1 but is rank-3
    allocate( test_array(7 : rank_3_array))

    ! Test that any comma list is parsed as AllocateShapeSpecList and not rewritten, 
    ! giving error messages expecting same number of 
    ! aruments as rank of test_array and scalar integers
    !ERROR: The number of shape specifications, when they appear, must match the rank of allocatable object
    !ERROR: Must be a scalar value, but is a rank-1 array
    !ERROR: Must be a scalar value, but is a rank-1 array
    !ERROR: Must be a scalar value, but is a rank-1 array
    !ERROR: Must have INTEGER type, but is REAL(4)
    allocate( test_array([1,2,3] : [2,3,4], 3, [1,2,3], 5.2))
    
  contains
    subroutine tmp02(unknown_size, test_ptr_01)
        real, allocatable, dimension(:,:,:), INTENT(OUT) :: test_ptr_01
        integer, INTENT(IN) :: unknown_size
        integer :: lower(unknown_size), upper(unknown_size)

        !ERROR: Rank-1 integer array used as upper bounds in ALLOCATE must have constant size
        allocate(test_ptr_01(upper))
        !ERROR: Rank-1 integer array used as lower bounds in ALLOCATE must have constant size
        !ERROR: Rank-1 integer array used as upper bounds in ALLOCATE must have constant size
        allocate(test_ptr_01(lower : upper))
        !ERROR: Rank-1 integer array used as lower bounds in ALLOCATE must have constant size
        allocate(test_ptr_01(lower : 10))
    end subroutine
 
end program
