! REQUIRES: openmp_runtime

! RUN: %python %S/../test_errors.py %s %flang_fc1 %openmp_flags
! Semantic checks on invalid atomic capture clause

use omp_lib
    logical x
    complex y
    !$omp atomic capture
    !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches operand types LOGICAL(4) and COMPLEX(4)
    x = y
    !ERROR: Operands of + must be numeric; have COMPLEX(4) and LOGICAL(4)
    y = y + x
    !$omp end atomic

    !$omp atomic capture
    !ERROR: Operands of + must be numeric; have COMPLEX(4) and LOGICAL(4)
    y = y + x
    !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches operand types LOGICAL(4) and COMPLEX(4)
    x = y
    !$omp end atomic
end
