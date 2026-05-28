! RUN: %python %S/../test_errors.py %s %flang_fc1
! REQUIRES: target=powerpc{{.*}}

subroutine test_vector_add()
  vector(integer(4)) :: v1, v2
  !ERROR: Operands of + must be numeric; have vector(integer(4)) and vector(integer(4))
  v1 = v1 + v2
end subroutine

subroutine test_vector_assignment()
  vector(integer(4)) :: v1
  vector(real(4)) :: v2
  !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches operand types vector(integer(4)) and vector(real(4))
  v1 = v2
end subroutine
