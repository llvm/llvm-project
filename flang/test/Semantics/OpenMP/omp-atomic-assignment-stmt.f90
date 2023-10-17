! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp
! Semantic checks for various assignments related to atomic constructs

program sample
    use omp_lib
    integer :: x, v
    integer :: y(10)
    integer, allocatable :: k
    integer a(10)
    type sample_type
        integer :: y
        integer :: m
    endtype
    type(sample_type) :: z
    !$omp atomic read
        v = x

    !$omp atomic read
    !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches scalar INTEGER(4) and rank 1 array of INTEGER(4)
    !ERROR: Expected scalar expression on the RHS of atomic assignment statement
        v = y(1:3)

    !$omp atomic read
    !ERROR: Expected scalar variable of intrinsic type on RHS of atomic assignment statement
        v = x * (10 + x)

    !$omp atomic read
    !ERROR: Expected scalar variable of intrinsic type on RHS of atomic assignment statement
        v = 4

    !$omp atomic read
    !ERROR: k must not have ALLOCATABLE attribute
        v = k

    !$omp atomic write
    !ERROR: k must not have ALLOCATABLE attribute
        k = x

    !$omp atomic update
    !ERROR: k must not have ALLOCATABLE attribute
        k = k + x * (v * x)

    !$omp atomic
    !ERROR: k must not have ALLOCATABLE attribute
        k = v * k  
         
    !$omp atomic write
    !ERROR: RHS expression on atomic assignment statement cannot access 'z%y'
       z%y = x + z%y

    !$omp atomic write
    !ERROR: RHS expression on atomic assignment statement cannot access 'x'
        x = x

    !$omp atomic write
    !ERROR: RHS expression on atomic assignment statement cannot access 'm'
        m = min(m, x, z%m) + k
 
    !$omp atomic read
    !ERROR: RHS expression on atomic assignment statement cannot access 'x'
        x = x

    !$omp atomic read
    !ERROR: Expected scalar variable of intrinsic type on RHS of atomic assignment statement
    !ERROR: RHS expression on atomic assignment statement cannot access 'm'
        m = min(m, x, z%m) + k

    !$omp atomic read
    !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches scalar INTEGER(4) and rank 1 array of INTEGER(4)
    !ERROR: Expected scalar expression on the RHS of atomic assignment statement
        x = a

    !$omp atomic read
    !ERROR: Expected scalar variable on the LHS of atomic assignment statement
        a = x

    !$omp atomic write
    !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches scalar INTEGER(4) and rank 1 array of INTEGER(4)
    !ERROR: Expected scalar expression on the RHS of atomic assignment statement
        x = a

    !$omp atomic write
    !ERROR: Expected scalar variable on the LHS of atomic assignment statement
        a = x
end program
