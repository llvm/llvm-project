!RUN: %flang_fc1 -fsyntax-only %s
!Regression test for crash in semantics on Cray pointers

module m
  pointer(ptr,pp)
end module m

program main
  use m, only:renamea=>pp
  use m, only:pp
  print *, renamea
end
