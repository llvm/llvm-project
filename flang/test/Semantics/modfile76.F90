!RUN: rm -rf %t && mkdir -p %t
!RUN: %flang_fc1 -fsyntax-only -fhermetic-module-files -DSTEP=1 -J%t %s
!RUN: %flang_fc1 -fsyntax-only -J%t %s

! Tests that a BIND(C) variable in a module A captured in a hermetic module
! file USE'd in a module B is not creating bogus complaints about BIND(C) name
! conflict when both module A and B are later accessed.

#if STEP == 1
module modfile76a
  integer, bind(c) :: x
end

module modfile76b
  use modfile76a ! capture hermetically
end

#else
subroutine test
  use modfile76a
  use modfile76b
  implicit none
  print *, x
end subroutine
#endif
