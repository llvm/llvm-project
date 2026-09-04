!RUN: rm -rf %t && mkdir -p %t
!RUN: %flang_fc1 -fsyntax-only -DWHICH=1 -J%t %s
!RUN: %flang_fc1 -fdebug-unparse -DWHICH=2 -J%t %s | FileCheck %s
! Empty typed named-constant arrays must keep their kind (and character
! length) across a module file into a separately compiled consumer.

#if WHICH == 1
module modfile84
  character(kind=4, len=0), parameter :: ec4(*) = [character(kind=4, len=0) ::]
  character(kind=2, len=5), parameter :: ec2(*) = [character(kind=2, len=5) ::]
  integer(2), parameter :: ei2(*) = [integer(2) ::]
  logical(8), parameter :: el8(*) = [logical(8) ::]
  real(2), parameter :: er2(*) = [real(2) ::]
end
#else
program test
  use modfile84
  !CHECK: PRINT *, 4_4, 0_4, 0_8
  print *, kind(ec4), len(ec4), size(ec4, kind=8)
  !CHECK: PRINT *, 2_4, 5_4
  print *, kind(ec2), len(ec2)
  !CHECK: PRINT *, 2_4
  print *, kind(ei2)
  !CHECK: PRINT *, 8_4
  print *, kind(el8)
  !CHECK: PRINT *, 2_4
  print *, kind(er2)
end
#endif
