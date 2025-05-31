! RUN: %python %S/../test_errors.py %s %flang -fopenacc

subroutine sub1(a)
 real, dimension(10) :: a
end subroutine

subroutine sub2(a)
  !$acc routine(sub1) gang(dim:1)
  real, dimension(10) :: a
  call sub1(a)
end subroutine

subroutine sub3()
  !$acc routine bind(sub1)
end subroutine

subroutine sub4()
  !ERROR: Only the dim argument is allowed on the GANG clause on the ROUTINE directive
  !$acc routine gang(num: 1)
end subroutine

subroutine sub5()
  !ERROR: Only the dim argument is allowed on the GANG clause on the ROUTINE directive
  !$acc routine gang(static: 1)
end subroutine

subroutine sub6()
  !ERROR: Clause GANG is not allowed if clause GANG appears on the ROUTINE directive
  !$acc routine gang gang

  !ERROR: Clause GANG is not allowed if clause WORKER appears on the ROUTINE directive
  !$acc routine worker gang

  !ERROR: Clause GANG is not allowed if clause VECTOR appears on the ROUTINE directive
  !$acc routine vector gang

  !ERROR: Clause GANG is not allowed if clause SEQ appears on the ROUTINE directive
  !$acc routine seq gang

  !ERROR: Clause WORKER is not allowed if clause WORKER appears on the ROUTINE directive
  !$acc routine worker worker

  !ERROR: Clause WORKER is not allowed if clause GANG appears on the ROUTINE directive
  !$acc routine gang worker

  !ERROR: Clause WORKER is not allowed if clause VECTOR appears on the ROUTINE directive
  !$acc routine vector worker

  !ERROR: Clause WORKER is not allowed if clause SEQ appears on the ROUTINE directive
  !$acc routine seq worker

  !ERROR: Clause VECTOR is not allowed if clause VECTOR appears on the ROUTINE directive
  !$acc routine vector vector

  !ERROR: Clause VECTOR is not allowed if clause GANG appears on the ROUTINE directive
  !$acc routine gang vector

  !ERROR: Clause VECTOR is not allowed if clause WORKER appears on the ROUTINE directive
  !$acc routine worker vector

  !ERROR: Clause VECTOR is not allowed if clause SEQ appears on the ROUTINE directive
  !$acc routine seq vector

  !ERROR: Clause SEQ is not allowed if clause SEQ appears on the ROUTINE directive
  !$acc routine seq seq

  !ERROR: Clause SEQ is not allowed if clause GANG appears on the ROUTINE directive
  !$acc routine gang seq

  !ERROR: Clause SEQ is not allowed if clause WORKER appears on the ROUTINE directive
  !$acc routine worker seq

  !ERROR: Clause SEQ is not allowed if clause VECTOR appears on the ROUTINE directive
  !$acc routine vector seq

end subroutine

subroutine sub7()
  !$acc routine device_type(*) gang device_type(host) worker

  !ERROR: Clause SEQ is not allowed if clause GANG appears on the ROUTINE directive
  !$acc routine device_type(*) gang seq

  !ERROR: Clause WORKER is not allowed if clause GANG appears on the ROUTINE directive
  !$acc routine device_type(*) gang worker

  !ERROR: Clause GANG is not allowed if clause GANG appears on the ROUTINE directive
  !$acc routine gang device_type(*) gang

  !ERROR: Clause WORKER is not allowed if clause GANG appears on the ROUTINE directive
  !$acc routine gang device_type(*) worker

  !ERROR: Clause VECTOR is not allowed if clause GANG appears on the ROUTINE directive
  !$acc routine gang device_type(*) vector

  !ERROR: Clause SEQ is not allowed if clause GANG appears on the ROUTINE directive
  !$acc routine gang device_type(*) seq

  !ERROR: Clause WORKER is not allowed if clause WORKER appears on the ROUTINE directive
  !$acc routine worker device_type(*) worker

  !ERROR: Clause GANG is not allowed if clause WORKER appears on the ROUTINE directive
  !$acc routine worker device_type(*) gang

  !ERROR: Clause VECTOR is not allowed if clause WORKER appears on the ROUTINE directive
  !$acc routine worker device_type(*) vector

  !ERROR: Clause SEQ is not allowed if clause WORKER appears on the ROUTINE directive
  !$acc routine worker device_type(*) seq

  !ERROR: Clause VECTOR is not allowed if clause VECTOR appears on the ROUTINE directive
  !$acc routine vector device_type(*) vector

  !ERROR: Clause GANG is not allowed if clause VECTOR appears on the ROUTINE directive
  !$acc routine vector device_type(*) gang

  !ERROR: Clause VECTOR is not allowed if clause VECTOR appears on the ROUTINE directive
  !$acc routine vector device_type(*) vector

  !ERROR: Clause SEQ is not allowed if clause VECTOR appears on the ROUTINE directive
  !$acc routine vector device_type(*) seq

  !ERROR: Clause SEQ is not allowed if clause SEQ appears on the ROUTINE directive
  !$acc routine seq device_type(*) seq

  !ERROR: Clause GANG is not allowed if clause SEQ appears on the ROUTINE directive
  !$acc routine seq device_type(*) gang

  !ERROR: Clause VECTOR is not allowed if clause SEQ appears on the ROUTINE directive
  !$acc routine seq device_type(*) vector

  !ERROR: Clause WORKER is not allowed if clause SEQ appears on the ROUTINE directive
  !$acc routine seq device_type(*) worker

  !$acc routine device_type(host) seq device_type(nvidia) gang device_type(multicore) vector device_type(*) worker
end subroutine
