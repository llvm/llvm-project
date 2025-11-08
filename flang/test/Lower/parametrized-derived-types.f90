! RUN: %python %S/test_errors.py %s %flang_fc1
! XFAIL: *
program main
  TYPE ty(k1,k2)
     INTEGER ,KIND::k1,k2=5
     INTEGER::arr(k1:k2)=10
     CHARACTER(LEN=k2)::CHARACTER
  END TYPE ty
  TYPE,EXTENDS(ty)::ty1(k3)
     INTEGER,KIND ::k3=4
     TYPE(ty(2,k3+1))::cmp_ty = ty(2,k3+1)(55,'HI')
  END TYPE ty1
  TYPE ty2(l1, l2)
  !ERROR: not yet implemented: parameterized derived types
     INTEGER,LEN ::l1,l2
     TYPE(ty1(2,5)), ALLOCATABLE::ty1_cmp(:)
  END TYPE ty2
  TYPE(ty2(4,8)) ::ty2_obj
end program main
