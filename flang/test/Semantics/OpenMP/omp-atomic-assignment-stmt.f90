! REQUIRES: openmp_runtime

! RUN: %python %S/../test_errors.py %s %flang_fc1 %openmp_flags
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

    !$omp atomic capture
        v = x
        x = x + 1
    !$omp end atomic

    !$omp atomic release capture
        v = x
    !ERROR: Atomic update statement should be of form `x = x operator expr` OR `x = expr operator x`
        x = b + (x*1)
    !$omp end atomic

    !$omp atomic capture hint(0)
        v = x
        x = 1
    !$omp end atomic

    !$omp atomic capture
    !ERROR: Captured variable/array element/derived-type component x expected to be assigned in the second statement of ATOMIC CAPTURE construct
        v = x
        b = b + 1
    !$omp end atomic

    !$omp atomic capture
    !ERROR: Captured variable/array element/derived-type component x expected to be assigned in the second statement of ATOMIC CAPTURE construct
        v = x
        b = 10
    !$omp end atomic

    !$omp atomic capture
    !ERROR: Updated variable/array element/derived-type component x expected to be captured in the second statement of ATOMIC CAPTURE construct
        x = x + 10
        v = b
    !$omp end atomic

    !$omp atomic capture
    !ERROR: Invalid ATOMIC CAPTURE construct statements. Expected one of [update-stmt, capture-stmt], [capture-stmt, update-stmt], or [capture-stmt, write-stmt]
        v = 1
        x = 4
    !$omp end atomic

    !$omp atomic capture
    !ERROR: Captured variable/array element/derived-type component z%y expected to be assigned in the second statement of ATOMIC CAPTURE construct
        x = z%y
        z%m = z%m + 1.0
    !$omp end atomic

    !$omp atomic capture
    !ERROR: Updated variable/array element/derived-type component z%m expected to be captured in the second statement of ATOMIC CAPTURE construct
        z%m = z%m + 1.0
        x = z%y
    !$omp end atomic

    !$omp atomic capture
    !ERROR: Captured variable/array element/derived-type component y(2) expected to be assigned in the second statement of ATOMIC CAPTURE construct
        x = y(2)
        y(1) = y(1) + 1
    !$omp end atomic

    !$omp atomic capture
    !ERROR: Updated variable/array element/derived-type component y(1) expected to be captured in the second statement of ATOMIC CAPTURE construct
        y(1) = y(1) + 1
        x = y(2)
    !$omp end atomic
end program
