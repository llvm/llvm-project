! RUN: %python %S/test_errors.py %s %flang_fc1
! Check errors on MAX/MIN with keywords, a weird case in Fortran
real :: x = 0.0, y = 0.0 , y1 = 0.0 ! prevent folding
!ERROR: Argument keyword 'a1=' was repeated in call to 'max'
print *, max(a1=x,a1=1)
!ERROR: Keyword argument 'a1=' has already been specified positionally (#1) in this procedure reference
print *, max(x,a1=1)
print *, max(a1=x,a2=0,a4=0) ! ok
print *, max(x,0,a99=0) ! ok
!ERROR: Argument keyword 'a06=' is not known in call to 'max'
print *, max(a1=x,a2=0,a06=0)
!ERROR: missing mandatory 'a2=' argument
print *, max(a3=y, a1=x)
!ERROR: missing mandatory 'a1=' argument
print *, max(a3=y, a2=x)
!ERROR: missing mandatory 'a1=' and 'a2=' arguments
print *, max(a3=y, a4=x)
!ERROR: missing mandatory 'a2=' argument
print *, max(y1, a3=y)
!ERROR: missing mandatory 'a1=' and 'a2=' arguments
print *, max(a9=x, a5=y, a4=y1)
!ERROR: missing mandatory 'a2=' argument
print *, max(x)
!ERROR: missing mandatory 'a2=' argument
print *, max(a1=x)
!ERROR: missing mandatory 'a1=' argument
print *, max(a2=y)
!ERROR: missing mandatory 'a1=' and 'a2=' arguments
print *, max(a3=x)
!ERROR: Argument keyword 'a0=' is not known in call to 'max'
print *, max(a0=x)
!ERROR: Argument keyword 'a03=' is not known in call to 'max'
print *, max(a03=x)
!ERROR: Argument keyword 'abad=' is not known in call to 'max'
print *, max(abad=x)
end
