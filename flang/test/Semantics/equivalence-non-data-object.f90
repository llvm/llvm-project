! RUN: %python %S/test_errors.py %s %flang_fc1

subroutine test3()
  common /blk/ k1
  ! ERROR: 'test3' in equivalence set is not a data object
  equivalence(i1, TEST3, k1) 
end subroutine test3
