! RUN: %python %S/test_errors.py %s %flang_fc1
! Test to check that this module can be compiled when an argument has no specified type
module inform
  implicit none
  interface normp
     module subroutine normt( array,n,val,p )
     integer siz
     integer,optional::p
     real*8 array(*)
     real*8 val
     end subroutine
  end interface
end

