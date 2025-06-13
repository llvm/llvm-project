!RUN: %flang_fc1 -fsyntax-only -fhermetic-module-files -DSTEP=1 %s
!RUN: %flang_fc1 -fsyntax-only %s

! Tests that a BIND(C) variable in a module A captured in a hermetic module
! file USE'd in a module B is not creating bogus complaints about BIND(C) name
! conflict when both module A and B are later accessed.

#if STEP == 1
module modfile75a
  integer, bind(c) :: x
end

module modfile75b
  use modfile75a ! capture hermetically
end

#else
subroutine test
  use modfile75a
  use modfile75b
  implicit none
  print *, x
end subroutine
#endif
