! REQUIRES: openmp_runtime

! RUN: %python %S/../test_errors.py %s %flang_fc1 %openmp_flags -fopenmp-version=50
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
    character :: l, r
    !$omp atomic read
        v = x

    !$omp atomic read
    !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches scalar INTEGER(4) and rank 1 array of INTEGER(4)
    !ERROR: Atomic variable y(1_8:3_8:1_8) should be a scalar
        v = y(1:3)

    !$omp atomic read
    !ERROR: Atomic expression x*(10_4+x) should be a variable
        v = x * (10 + x)

    !$omp atomic read
    !ERROR: Atomic expression 4_4 should be a variable
        v = 4

    !$omp atomic read
    !ERROR: Atomic variable k cannot be ALLOCATABLE
        v = k

    !$omp atomic write
    !ERROR: Atomic variable k cannot be ALLOCATABLE
        k = x

    !$omp atomic update
    !ERROR: Atomic variable k cannot be ALLOCATABLE
        k = k + x * (v * x)

    !$omp atomic
    !ERROR: Atomic variable k cannot be ALLOCATABLE
        k = v * k  
         
    !$omp atomic write
    !ERROR: Within atomic operation z%y and x+z%y access the same storage
       z%y = x + z%y

    !$omp atomic write
    !ERROR: Within atomic operation x and x access the same storage
        x = x

    !$omp atomic write
    !ERROR: Within atomic operation m and min(m,x,z%m)+k access the same storage
        m = min(m, x, z%m) + k
 
    !$omp atomic read
    !ERROR: Within atomic operation x and x access the same storage
        x = x

    !$omp atomic read
    !ERROR: Atomic expression min(m,x,z%m)+k should be a variable
        m = min(m, x, z%m) + k

    !$omp atomic read
    !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches scalar INTEGER(4) and rank 1 array of INTEGER(4)
    !ERROR: Atomic variable a should be a scalar
        x = a

    !$omp atomic write
    !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches scalar INTEGER(4) and rank 1 array of INTEGER(4)
        x = a

    !$omp atomic write
    !ERROR: Atomic variable a should be a scalar
        a = x

    !$omp atomic capture
        v = x
        x = x + 1
    !$omp end atomic

    !$omp atomic release capture
        v = x
    !ERROR: An implicit or explicit type conversion is not a valid ATOMIC UPDATE operation
        x = b + (x*1)
    !$omp end atomic

    !$omp atomic capture hint(0)
        v = x
        x = 1
    !$omp end atomic

    !$omp atomic capture
        v = x
    !ERROR: In ATOMIC UPDATE operation with CAPTURE the update statement should assign to x
        b = b + 1
    !$omp end atomic

    !$omp atomic capture
        v = x
    !ERROR: In ATOMIC UPDATE operation with CAPTURE the update statement should assign to x
        b = 10
    !$omp end atomic

    !ERROR: Unable to identify capture statement: in ATOMIC UPDATE operation with CAPTURE the source value in the capture statement should be a variable (with the same type as the target)
    !$omp atomic capture
        x = x + 10
        v = b
    !$omp end atomic

    !ERROR: Unable to identify capture statement: in ATOMIC UPDATE operation with CAPTURE the source value in the capture statement should be a variable (with the same type as the target)
    !$omp atomic capture
        v = 1
        x = 4
    !$omp end atomic

    !$omp atomic capture
        x = z%y
    !ERROR: In ATOMIC UPDATE operation with CAPTURE the update statement should assign to z%y
        z%m = z%m + 1.0
    !$omp end atomic

    !$omp atomic capture
    !ERROR: In ATOMIC UPDATE operation with CAPTURE the update statement should assign to z%y
        z%m = z%m + 1.0
        x = z%y
    !$omp end atomic

    !$omp atomic capture
        x = y(2)
    !ERROR: In ATOMIC UPDATE operation with CAPTURE the update statement should assign to y(2_8)
        y(1) = y(1) + 1
    !$omp end atomic

    !$omp atomic capture
    !ERROR: In ATOMIC UPDATE operation with CAPTURE the update statement should assign to y(2_8)
        y(1) = y(1) + 1
        x = y(2)
    !$omp end atomic

    !$omp atomic read
    !ERROR: Atomic variable r cannot have CHARACTER type
        l = r

    !$omp atomic write
    !ERROR: Atomic variable l cannot have CHARACTER type
        l = r
end program
