! RUN: %python %S/test_errors.py %s %flang_fc1
! Regression test for bug #119151
interface sub
  subroutine sub1(ifun)
    interface
      integer function ifun()
      end
     end interface
   end
   subroutine sub2(rfun)
     real rfun
     external rfun
   end
end interface
integer ifun
real rfun
complex zfun
external ifun, rfun, zfun, xfun
call sub(ifun)
call sub(rfun)
!ERROR: No specific subroutine of generic 'sub' matches the actual arguments
call sub(zfun)
call sub(xfun)
end
