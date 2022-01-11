! RUN: %python %S/test_errors.py %s %flang_fc1
! Check errors on MAX/MIN with keywords, a weird case in Fortran
real :: x = 0.0 ! prevent folding
!ERROR: Argument keyword 'a1=' was repeated in call to 'max'
print *, max(a1=x,a1=1)
!ERROR: Keyword argument 'a1=' has already been specified positionally (#1) in this procedure reference
print *, max(x,a1=1)
print *, max(a1=x,a2=0,a4=0) ! ok
print *, max(x,0,a99=0) ! ok
!ERROR: Argument keyword 'a06=' is not known in call to 'max'
print *, max(a1=x,a2=0,a06=0)
end
