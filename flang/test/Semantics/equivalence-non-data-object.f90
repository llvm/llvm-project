! RUN: %python %S/test_errors.py %s %flang_fc1

subroutine test3()
  common /blk/ k1
  ! ERROR: 'test3' in equivalence set is not a data object
  equivalence(i1, TEST3, k1) 
end subroutine test3

subroutine test4()
  common a
  ! ERROR: 'test4' in equivalence set is not a data object
  equivalence(a, test4)
end subroutine test4

module m
  real val
end

  use m
  common a
  ! ERROR: Use-associated variable 'val' is not allowed in an equivalence set
  equivalence(a, val)
end
