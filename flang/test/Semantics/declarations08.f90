! RUN: %python %S/test_errors.py %s %flang_fc1
pointer(p,x)
!ERROR: Cray pointee 'y' may not be a member of an EQUIVALENCE group
pointer(p,y)
!ERROR: Cray pointee 'x' may not be a member of COMMON block //
common x
equivalence(y,z)
!ERROR: Cray pointee 'v' may not be initialized
real :: v = 42.0
pointer(p,v)
!ERROR: Cray pointee 'u' may not have the SAVE attribute
save u
pointer(p, u)
end
