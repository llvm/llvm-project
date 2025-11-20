! RUN: %python %S/test_errors.py %s %flang_fc1
! Check errors on MAX/MIN with keywords, a weird case in Fortran
real :: x = 0.0, y = 0.0 , y1 = 0.0 ! prevent folding
!ERROR: argument keyword 'a1=' was repeated in call to 'max'
print *, max(a1=x,a1=1)
!ERROR: keyword argument 'a1=' to intrinsic 'max' was supplied positionally by an earlier actual argument
print *, max(x,a1=1)
!ERROR: keyword argument 'a1=' to intrinsic 'min1' was supplied positionally by an earlier actual argument
print *, min1(1.,a1=2.,a2=3.)
print *, max(a1=x,a2=0,a4=0) ! ok
print *, max(x,0,a99=0) ! ok
!ERROR: argument keyword 'a06=' is not known in call to 'max'
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
!ERROR: argument keyword 'a0=' is not known in call to 'max'
print *, max(a0=x)
!ERROR: argument keyword 'a03=' is not known in call to 'max'
print *, max(a03=x)
!ERROR: argument keyword 'abad=' is not known in call to 'max'
print *, max(abad=x)
!ERROR: actual argument #2 without a keyword may not follow an actual argument with a keyword
print *, max(a1=x, y)
!ERROR: Keyword argument 'a3=' has already been specified positionally (#3) in this procedure reference
print *, max(x, y, y1, a3=x)
print *, max(a3=x, a4=x, a2=x, a1=x) ! ok
print *, max(a3=x, a2=y, a4=x, a1=x) ! ok
print *, max(x, a2=y, a5=x, a4=x) ! ok
print *, max(x, y, a4=x, a6=x) ! ok
end
