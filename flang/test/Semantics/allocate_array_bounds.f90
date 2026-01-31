! RUN: %python %S/test_errors.py %s %flang_fc1
program int_array_alloc_03
    implicit none
    real, allocatable, dimension(:,:,:) :: test_array_01
    real, allocatable, dimension(:,:,:) :: test_array_02
    real, allocatable, dimension(:,:,:) :: test_array_03
    
    integer :: lower(4), upper(4)
    
    !ERROR: ALLOCATE bounds arrays must have the same size; lower bounds has 3 elements, upper bounds has 2 elements
    allocate( test_array_01([1,2,3]:[3,3]))
    !ERROR: Must have INTEGER type, but is REAL(4)
    allocate( test_array_02([1.2,2.2,3.2]:[1,2,3]))
    !ERROR: ALLOCATE bounds array has 4 elements but allocatable object 'test_array_03' has rank 3
    allocate( test_array_03(lower:upper) )

!   contains
!     subroutine tmp02(unknown_size, test_ptr_01)
!         real, allocatable, dimension(:,:,:), INTENT(OUT) :: test_ptr_01
!         integer, INTENT(IN) :: unknown_size
!         integer :: upper(unknown_size)

!         allocate(test_ptr_01(upper))
!     end subroutine
 
end program
