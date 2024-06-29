! RUN: %python %S/test_errors.py %s %flang_fc1
pointer(p,x)
!ERROR: Cray pointee 'y' may not be a member of an EQUIVALENCE group
pointer(p,y)
!ERROR: Cray pointee 'x' may not be a member of a COMMON block
common x
equivalence(y,z)
end
