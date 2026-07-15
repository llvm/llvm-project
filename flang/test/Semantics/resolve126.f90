! RUN: %python %S/test_errors.py %s %flang_fc1
!ERROR: Attributes 'INTRINSIC' and 'EXTERNAL' conflict with each other
real, external, intrinsic :: exp
!ERROR: Symbol 'sin' cannot have both EXTERNAL and INTRINSIC attributes
external sin
intrinsic sin
end
